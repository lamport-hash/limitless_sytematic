from typing import Dict, List, Tuple
from datetime import datetime
import pickle
import json
import os
import asyncio
import logging
import yaml
from pathlib import Path
import inspect
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks, Depends

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.routing import APIRoute

from core.classes import (
    NormalizedOrderBook,
    NormalizedTrade,
    NormalizedCandle,
)
from .listener_redis_client import RedisClient, MktDataType
from .config import get_all_listeners_config
from .portal_client import (
    router,
    get_db,
    get_user_status_html,
    verify_portal_service_token,
    SERVICE_NAME,
)
from sqlalchemy.orm import Session

templates = Jinja2Templates(directory="templates")

app = FastAPI()
app.include_router(router, prefix="/clientportal")

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_signature(signature: str) -> bool:
    """Placeholder for signature validation logic."""
    # Replace with your actual signature verification code.
    return True  # Or False if invalid


redis_client = RedisClient()  # Initialize redis client outside the endpoints

# Global variable to store all listeners configuration
listeners_config: Dict[str, Dict] = {}


async def update_listeners_config():
    """Update the global listeners configuration from YAML file"""
    global listeners_config
    try:
        listeners_config = await asyncio.get_event_loop().run_in_executor(
            None, get_all_listeners_config
        )
        logger.info(f"Updated listeners config with {len(listeners_config)} listeners")
    except Exception as e:
        logger.error(f"Failed to update listeners config: {e}")


@app.on_event("startup")
async def startup_event():
    # Initial load of listeners config
    await update_listeners_config()

    # Start background task to update config every 5 minutes
    asyncio.create_task(config_update_loop())


async def config_update_loop():
    """Background loop to update listeners config every 5 minutes"""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        await update_listeners_config()


@app.get("/last_orderbook")
async def get_last_orderbook(
    exchange_name: str,
    symbol: str,
    listener_id: str = Query(..., description="Listener ID"),
    json_output: bool = True,
    signature: str = Query(...),
):
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")

    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    orderbook_data = redis_client.get_market_data(
        listener_id, connector, exchange_name, symbol, MktDataType.ORDERBOOK
    )
    if orderbook_data:
        if json_output:
            return orderbook_data.to_json()  # Convert dict to object
        else:
            return pickle.dumps(orderbook_data)
    else:
        raise HTTPException(status_code=404, detail="Orderbook not found")


@app.get("/last_n_trades")
async def get_last_n_trades(
    exchange_name: str,
    symbol: str,
    listener_id: str = Query(..., description="Listener ID"),
    n: int = 10,
    json_output: bool = True,
    signature: str = Query(...),
):
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")

    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    trades_data = redis_client.get_market_data(
        listener_id, connector, exchange_name, symbol, MktDataType.TRADE, n
    )
    if trades_data:
        if json_output:
            return [
                trade.to_json() for trade in trades_data
            ]  # Convert list of dicts to list of objects
        else:
            return pickle.dumps(trades_data)
    else:
        raise HTTPException(status_code=404, detail="Trades not found")


@app.get("/candles/last_n_candles")
async def get_last_n_candles(
    exchange_name: str,
    symbol: str,
    listener_id: str = Query(..., description="Listener ID"),
    n: int = 10,
    timeframe: str = "1m",
    json_output: bool = True,
    signature: str = Query(...),
):
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")

    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    # candles_data = redis_client.get_candles(exchange_name, symbol, timeframe)
    candles_data = redis_client.get_candles(
        listener_id, connector, exchange_name, symbol, n
    )

    if candles_data:
        if json_output:
            return [
                candle.to_json() for candle in candles_data
            ]  # Convert list of dicts to list of objects
        else:
            return pickle.dumps(candles_data)
    else:
        raise HTTPException(status_code=404, detail="Candles not found")


def get_alive_listeners_for_symbol(symbol: str) -> List[Tuple[str, str]]:
    """Get list of (listener_id, exchange_name) that are alive for the symbol"""
    alive = []
    current_time = datetime.now().timestamp() * 1000  # in ms
    threshold = 3600000  # 1 hour in ms

    symbols = redis_client.get_all_symbols()
    for exchange, sym, listener_id, connector_type in symbols:
        if sym == symbol and listener_id in listeners_config:
            ts = redis_client.get_last_update_ts(
                listener_id, connector_type, exchange, sym
            )
            if ts and (current_time - ts) < threshold:
                alive.append((listener_id, exchange))
    return alive


def load_priority_config() -> Dict[str, List[Dict[str, str]]]:
    """Load priority config from yaml file"""
    if not PRIORITY_CONFIG_PATH.exists():
        return {}
    try:
        with open(PRIORITY_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML priority config file: {e}")
        return {}


@app.get("/candles/last_n_candles_default")
async def last_n_candles_default(
    symbol: str,
    n: int = 10,
    timeframe: str = "1m",
    json_output: bool = True,
    signature: str = Query(...),
):
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")

    alive_listeners = get_alive_listeners_for_symbol(symbol)
    if not alive_listeners:
        raise HTTPException(
            status_code=404, detail=f"No alive listeners for symbol {symbol}"
        )

    selected_listener_id = None
    selected_exchange = None

    if len(alive_listeners) == 1:
        selected_listener_id, selected_exchange = alive_listeners[0]
    else:
        # Load priority config
        priority_config = load_priority_config()
        if symbol in priority_config:
            priorities = priority_config[symbol]
            for pri in priorities:
                exchange = pri.get("exchange")
                listener = pri.get("listener")
                if (listener, exchange) in alive_listeners:
                    selected_listener_id = listener
                    selected_exchange = exchange
                    break
        if not selected_listener_id:
            # If no priority match, pick the first alive
            selected_listener_id, selected_exchange = alive_listeners[0]

    # Now call get_last_n_candles with the selected ones
    # But since it's async, I need to call it directly, not via the route
    if selected_listener_id not in listeners_config:
        raise HTTPException(
            status_code=404,
            detail=f"Selected listener {selected_listener_id} not found",
        )

    assert selected_exchange is not None
    connector = listeners_config[selected_listener_id]["connector"]
    candles_data = redis_client.get_candles(
        selected_listener_id, connector, selected_exchange, symbol, n
    )

    connector = listeners_config[selected_listener_id]["connector"]
    candles_data = redis_client.get_candles(
        selected_listener_id, connector, selected_exchange, symbol, n
    )

    if candles_data:
        if json_output:
            return [candle.to_json() for candle in candles_data]
        else:
            return pickle.dumps(candles_data)
    else:
        raise HTTPException(status_code=404, detail="Candles not found")


def is_today(day: datetime) -> bool:
    return day.date() == datetime.today().date()


@app.get("/candles/all_daily_candles")
async def get_all_1min_candles_for_day(
    exchange_name: str,
    symbol: str,
    day: datetime,
    listener_id: str = Query(..., description="Listener ID"),
    json_output: bool = True,
    signature: str = Query(...),
):
    # logger.error(f"Received request - exchange: {exchange_name}, symbol: {symbol}, day: {day}, json_output{json_output}")
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")

    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    # logger.error(f"fetching candles from redis" )

    candles_data = redis_client.get_daily_candles(
        listener_id, connector, exchange_name, symbol, day, is_today(day)
    )
    if candles_data:
        # logger.error(f"candles {len(candles_data)}" )
        if json_output:
            return [
                candle.to_json() for candle in candles_data
            ]  # Convert list of dicts to list of objects
        else:
            return pickle.dumps(candles_data)
    else:
        raise HTTPException(status_code=404, detail="Candles not found")


@app.get("/candles/count")
def get_nb_candles(
    exchange: str = Query(...),
    symbol: str = Query(...),
    listener_id: str = Query(..., description="Listener ID"),
    date: str = Query(..., regex=r"^\d{4}/\d{2}/\d{2}$"),
) -> int:
    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    return redis_client.get_nb_candles(listener_id, connector, exchange, symbol, date)


@app.get("/candles/daily_counts")
def get_daily_candle_counts(
    exchange: str = Query(...),
    symbol: str = Query(...),
    listener_id: str = Query(..., description="Listener ID"),
    n_days: int = Query(7, gt=0),
) -> Dict[str, int]:
    if listener_id not in listeners_config:
        raise HTTPException(status_code=404, detail=f"Listener {listener_id} not found")

    connector = listeners_config[listener_id]["connector"]
    # Return as dict (FastAPI will convert OrderedDict fine)
    return redis_client.get_daily_candle_counts(
        listener_id, connector, exchange, symbol, n_days
    )


@app.get("/list_symbols")
def list_symbols():
    """List all currently monitored symbols"""
    return ""


@app.get("/active_listeners")
async def get_active_listeners():
    """Get list of all configured listeners"""
    if not listeners_config:
        await update_listeners_config()
    return listeners_config


@app.get("/redis/stats")
def get_redis_stats(
    listener_id: str = Query(None, description="Listener ID (optional for totals)"),
):
    """Get Redis database statistics"""
    if listener_id:
        if listener_id not in listeners_config:
            raise HTTPException(
                status_code=404, detail=f"Listener {listener_id} not found"
            )
        connector = listeners_config[listener_id]["connector"]
        return {
            "candles_count": redis_client.get_total_count(
                listener_id, connector, MktDataType.CANDLE
            ),
            "trades_count": redis_client.get_total_count(
                listener_id, connector, MktDataType.TRADE
            ),
            "orderbooks_count": redis_client.get_total_count(
                listener_id, connector, MktDataType.ORDERBOOK
            ),
            "memory_usage": redis_client.get_redis_memory_usage(),
        }
    else:
        # Return totals across all listeners
        total_candles = 0
        total_trades = 0
        total_orderbooks = 0
        for lid, config in listeners_config.items():
            connector = config["connector"]
            total_candles += redis_client.get_total_count(
                lid, connector, MktDataType.CANDLE
            )
            total_trades += redis_client.get_total_count(
                lid, connector, MktDataType.TRADE
            )
            total_orderbooks += redis_client.get_total_count(
                lid, connector, MktDataType.ORDERBOOK
            )
        return {
            "candles_count": total_candles,
            "trades_count": total_trades,
            "orderbooks_count": total_orderbooks,
            "memory_usage": redis_client.get_redis_memory_usage(),
        }


@app.get("/listener_ports")
async def get_listener_ports():
    # Return configured listeners with their ports
    result = {}
    symbols = redis_client.get_all_symbols()
    for listener_id, config in listeners_config.items():
        # Find max last_update_ts for this listener's symbols
        max_ts = None
        for exchange, symbol, lid, connector_type in symbols:
            if lid == listener_id:
                ts = redis_client.get_last_update_ts(
                    listener_id, connector_type, exchange, symbol
                )
                if ts and (max_ts is None or ts > max_ts):
                    max_ts = ts
        last_seen = (
            datetime.fromtimestamp(max_ts / 1000).isoformat() if max_ts else None
        )
        result[listener_id] = {
            "port": config.get("port"),
            "connector": config.get("connector"),
            "historize": config.get("historize", False),
            "datetime": last_seen,
        }
    return result


@app.get("/monitoring/data")
async def get_monitoring_data(signature: str = Query(...)):
    if not validate_signature(signature):
        raise HTTPException(status_code=401, detail="Invalid Signature")
    data = redis_client.get_monitoring_data()
    return data


@app.get("/manager_listener_ui", response_class=HTMLResponse)
async def get_manager_listener_ui(
    request: Request, user_data: dict = Depends(verify_portal_service_token)
):
    user_logon_message = {
        "message": "Token is valid",
        "user_info": {
            "user_id": user_data.get("user_id"),
            "email": user_data.get("email"),
        },
        "service": SERVICE_NAME,
    }
    # Return the HTML content above
    data = redis_client.get_monitoring_data()
    return templates.TemplateResponse(
        "manager_listener_ui.html",
        {
            "request": request,
            "monitoring_data": data,
            "user_logon_message": user_logon_message,
            "active_listeners": listeners_config,
            "current_listener_id": None,  # No specific listener for manager
            "connector_type": None,  # No specific connector for manager
        },
    )

    # Return the HTML content above
    data = redis_client.get_monitoring_data()
    return templates.TemplateResponse(
        "manager_listener_ui.html",
        {
            "request": request,
            "monitoring_data": data,
            "active_listeners": listeners_config,
            "current_listener_id": None,  # No specific listener for manager
            "connector_type": None,  # No specific connector for manager
        },
    )


from fastapi.routing import APIRoute

for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.path} -> {', '.join(route.methods)}")


service_name = "Listener Manager"


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request, token: Optional[str] = Query(None), db: Session = Depends(get_db)
):
    # if not token:
    #    raise HTTPException(status_code=401, detail="Token required")
    ## Get user status HTML from the endpoint
    user_status_html = get_user_status_html(db)

    # Collect all endpoints dynamically
    endpoints = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            protected = False
            sig = inspect.signature(route.endpoint)
            if "signature" in sig.parameters or any(
                dep.dependency == get_db for dep in route.dependencies
            ):
                protected = True
            endpoints.append(
                {
                    "path": route.path,
                    "methods": list(route.methods),
                    "protected": protected,
                }
            )

    # Build endpoints HTML
    endpoints_html = ""
    for ep in endpoints:
        color = "purple" if ep["protected"] else "gray"
        methods_str = ", ".join(ep["methods"])
        protected_str = "Protected" if ep["protected"] else "Public"
        endpoints_html += f"""
        <div class="p-4 bg-{color}-50 dark:bg-{color}-900 rounded-lg">
            <code class="text-{color}-900 dark:text-{color}-100 font-mono">{methods_str} {ep["path"]}</code>
            <span class="text-{color}-700 dark:text-{color}-300 ml-2">{protected_str}</span>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{service_name} Service Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 dark:bg-gray-900">
        <div class="min-h-screen flex flex-col">
            <!-- Header -->
            <header class="bg-white dark:bg-gray-800 shadow-sm border-b">
                <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div class="flex justify-between items-center py-4">
                        <div>
                            <h1 class="text-2xl font-bold text-blue-600 dark:text-blue-400">{service_name} Service</h1>
                            <p class="text-gray-600 dark:text-gray-300">Microservice Dashboard</p>
                        </div>
                        {user_status_html}
                    </div>
                </div>
            </header>

            <!-- Main Content -->
            <main class="flex-1 py-8">
                <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                     <!-- Service Info Card -->
                     <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 mb-8">
                          <h2 class="text-3xl font-bold mb-4 text-gray-900 dark:text-white">{service_name} Service Dashboard</h2>
                         <p class="text-gray-600 dark:text-gray-300 mb-6">This service provides analytics functionality for the microservice ecosystem.</p>
                        
                         <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                             <div class="bg-blue-50 dark:bg-blue-900 p-6 rounded-lg">
                                 <h3 class="font-semibold text-blue-900 dark:text-blue-100 mb-2">Service Status</h3>
                                 <p class="text-blue-700 dark:text-blue-300">Running and healthy</p>
                             </div>
                             <div class="bg-green-50 dark:bg-green-900 p-6 rounded-lg">
                                 <h3 class="font-semibold text-green-900 dark:text-green-100 mb-2">Authentication</h3>
                                 <p class="text-green-700 dark:text-green-300">Token-based validation enabled</p>
                             </div>
                         </div>

                         <!-- Token Test Section -->
                         <div class="border-t dark:border-gray-600 pt-6">
                             <h3 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Token Validation Test</h3>
                             <p class="text-gray-600 dark:text-gray-300 mb-4">Test the protected endpoint with your authentication token:</p>
                            
                              <div class="flex flex-col sm:flex-row gap-4">
                                  <button id="test-token-btn"
                                          class="bg-purple-500 hover:bg-purple-700 text-white font-bold py-3 px-6 rounded-lg transition flex-1">
                                      Test Protected Route
                                  </button>
                                  <button id="health-check-btn"
                                          class="bg-gray-500 hover:bg-gray-700 text-white font-bold py-3 px-6 rounded-lg transition flex-1">
                                      Health Check
                                  </button>
                                  <button id="open-manager-ui-btn"
                                          class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition flex-1">
                                      Open Manager Listener UI
                                  </button>
                              </div>

                              <!-- Results Display -->
                              <div id="test-results" class="hidden mt-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                  <h4 class="font-semibold mb-2 text-gray-900 dark:text-white">Test Results:</h4>
                                  <pre id="results-content" class="text-sm overflow-auto text-gray-900 dark:text-gray-100"></pre>
                              </div>

                              <!-- Health Check Results Display -->
                              <div id="health-results" class="hidden mt-6 p-4 bg-green-50 dark:bg-green-900 rounded-lg">
                                  <h4 class="font-semibold mb-2 text-green-900 dark:text-green-100">Health Check Results:</h4>
                                  <pre id="health-results-content" class="text-sm overflow-auto text-green-700 dark:text-green-300"></pre>
                              </div>
                        </div>
                    </div>

                      <!-- API Endpoints -->
                      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
                          <h3 class="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Available Endpoints</h3>
                          <div class="grid gap-4">
                              {endpoints_html}
                          </div>
                      </div>
                </div>
            </main>

             <!-- Footer -->
             <footer class="bg-white dark:bg-gray-800 border-t dark:border-gray-600 py-4">
                 <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-500 dark:text-gray-400">
                     <p>PortalClient Service © 2024</p>
                 </div>
             </footer>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', async function() {{
                const testBtn = document.getElementById('test-token-btn');
                const resultsDiv = document.getElementById('test-results');
                const resultsContent = document.getElementById('results-content');

                const healthBtn = document.getElementById('health-check-btn');
                const healthResultsDiv = document.getElementById('health-results');
                const healthResultsContent = document.getElementById('health-results-content');

                testBtn.addEventListener('click', async function() {{
                    testBtn.disabled = true;
                    testBtn.textContent = 'Testing...';

                    try {{
                        // Try to get token from various sources
                        let token = null;

                        // First check localStorage (from auto-login)
                        token = localStorage.getItem('service_token');

                        // Fallback to URL param
                        if (!token) {{
                            const urlParams = new URLSearchParams(window.location.search);
                            token = urlParams.get('token');
                        }}

                        const response = await fetch('/clientportal/protected_by_service_token', {{
                            method: 'GET',
                            headers: {{
                                'Authorization': token ? `Bearer ${{token}}` : ''
                            }}
                        }});

                        const data = await response.json();

                        resultsDiv.classList.remove('hidden');
                        resultsContent.textContent = JSON.stringify({{
                            status: response.status,
                            statusText: response.statusText,
                            data: data
                        }}, null, 2);

                        if (response.ok) {{
                            resultsContent.className = 'text-green-700';
                        }} else {{
                            resultsContent.className = 'text-red-700';
                        }}

                    }} catch (error) {{
                        resultsDiv.classList.remove('hidden');
                        resultsContent.className = 'text-red-700';
                        resultsContent.textContent = JSON.stringify({{
                            error: 'Network error',
                            message: error.message
                        }}, null, 2);
                    }} finally {{
                        testBtn.disabled = false;
                        testBtn.textContent = 'Test Protected Route';
                    }}
                }});

                 // Health check button handler
                healthBtn.addEventListener('click', async function() {{
                    healthBtn.disabled = true;
                    healthBtn.textContent = 'Checking...';

                    try {{
                        const response = await fetch('/clientportal/health');
                        const data = await response.json();

                        healthResultsDiv.classList.remove('hidden');
                        healthResultsContent.textContent = JSON.stringify({{
                            status: response.status,
                            statusText: response.statusText,
                            data: data
                        }}, null, 2);

                        if (response.ok) {{
                            healthResultsContent.className = 'text-green-700';
                        }} else {{
                            healthResultsContent.className = 'text-red-700';
                        }}

                    }} catch (error) {{
                        healthResultsDiv.classList.remove('hidden');
                        healthResultsContent.className = 'text-red-700';
                        healthResultsContent.textContent = JSON.stringify({{
                            error: 'Network error',
                            message: error.message
                        }}, null, 2);
                    }} finally {{
                        healthBtn.disabled = false;
                        healthBtn.textContent = 'Health Check';
                    }}
                }});

                // Open Manager Listener UI button handler
                const openManagerBtn = document.getElementById('open-manager-ui-btn');
                openManagerBtn.addEventListener('click', async function() {{
                    // Try to get token from various sources
                    let token = null;

                    // First check localStorage (from auto-login)
                    token = localStorage.getItem('service_token');

                    // Fallback to URL param
                    if (!token) {{
                        const urlParams = new URLSearchParams(window.location.search);
                        token = urlParams.get('token');
                    }}

                    if (token) {{
                        try {{
                            const response = await fetch('/manager_listener_ui', {{
                                method: 'GET',
                                headers: {{
                                    'Authorization': `Bearer ${{token}}`
                                }}
                            }});
                            if (response.ok) {{
                                const html = await response.text();
                                document.body.innerHTML = html;
                            }} else {{
                                const errorData = await response.json();
                                alert('Error: ' + (errorData.detail || 'Unknown error'));
                            }}
                        }} catch (error) {{
                            alert('Network error: ' + error.message);
                        }}
                    }} else {{
                        alert('No authentication token found. Please log in first.');
                    }}
                }});

                // Auto-login with token from URL
                const urlParams = new URLSearchParams(window.location.search);
                const tokenParam = urlParams.get('token');
                if (tokenParam) {{
                    console.log('Found token in URL, attempting auto-login...');

                    try {{
                        // First verify the token to get user info
                                                    const verifyResponse = await fetch('/clientportal/protected_by_service_token', {{
                            method: 'GET',
                            headers: {{
                                'Authorization': `Bearer ${{tokenParam}}`
                            }}
                        }});

                        if (verifyResponse.ok) {{
                            const verifyData = await verifyResponse.json();
                            const userInfo = verifyData.user_info;

                            // Now login with the user data
                            const loginPayload = {{
                                user_data: {{
                                    email: userInfo.email,
                                    full_name: userInfo.email, // Use email as fallback for name
                                    user_id: userInfo.user_id
                                }},
                                token: tokenParam
                            }};

                            const loginResponse = await fetch('/clientportal/login_with_portal_token', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json'
                                }},
                                body: JSON.stringify(loginPayload)
                            }});

                             if (loginResponse.ok) {{
                                 const data = await loginResponse.json();
                                 console.log('Auto-login successful');
                                 // Store token for future use (e.g., testing protected routes)
                                 localStorage.setItem('service_token', tokenParam);
                                 // Apply theme
                                 if (data.dark_mode) {{
                                     document.documentElement.classList.add('dark');
                                 }}
                                 // Remove token from URL and refresh to show logged in status
                                 const url = new URL(window.location);
                                 url.searchParams.delete('token');
                                 window.location.href = url.toString();
                             }} else {{
                                 console.error('Auto-login failed:', loginResponse.status);
                             }}
                        }} else {{
                            console.error('Token verification failed:', verifyResponse.status);
                        }}
                    }} catch (error) {{
                        console.error('Auto-login error:', error);
                    }}
                }}

                // Logout button handler
                const logoutBtn = document.getElementById('logout-btn');
                if (logoutBtn) {{
                    logoutBtn.addEventListener('click', async function() {{
                        // Get token from localStorage or URL
                        let token = localStorage.getItem('service_token');
                        if (!token) {{
                            const urlParams = new URLSearchParams(window.location.search);
                            token = urlParams.get('token');
                        }}

                        if (token) {{
                            try {{
                                const response = await fetch('/clientportal/logout', {{
                                    method: 'POST',
                                    headers: {{
                                        'Authorization': `Bearer ${{token}}`
                                    }}
                                }});

                                if (response.ok) {{
                                    // Clear stored token and redirect
                                    localStorage.removeItem('service_token');
                                    const url = new URL(window.location);
                                    url.searchParams.delete('token');
                                    window.location.href = url.toString();
                                }} else {{
                                    const errorData = await response.json();
                                    alert('Logout failed: ' + (errorData.detail || 'Unknown error'));
                                }}
                            }} catch (error) {{
                                console.error('Logout error:', error);
                                alert('Network error during logout.');
                            }}
                        }}
                    }});
                }}

                // Register button handler
                const registerBtn = document.getElementById('register-btn');
                if (registerBtn) {{
                    registerBtn.addEventListener('click', async function() {{
                        registerBtn.disabled = true;
                        registerBtn.textContent = 'Registering...';

                        try {{
                            const response = await fetch('/clientportal/register');
                            const data = await response.json();

                            const resultSpan = document.getElementById('register-result');
                            resultSpan.textContent = data.message || JSON.stringify(data);
                            resultSpan.className = response.ok ? 'text-green-600 text-sm' : 'text-red-600 text-sm';
                        }} catch (error) {{
                            const resultSpan = document.getElementById('register-result');
                            resultSpan.textContent = 'Error: ' + error.message;
                            resultSpan.className = 'text-red-600 text-sm';
                        }} finally {{
                            registerBtn.disabled = false;
                            registerBtn.textContent = 'Register Service';
                        }}
                    }});
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
