document.addEventListener('DOMContentLoaded', function() {
    const symbolsContainer = document.getElementById('symbols-container');
    const refreshBtn = document.getElementById('refresh-btn');
    const refreshIntervalInput = document.getElementById('refresh-interval');
    const connectionStatus = document.getElementById('connection-status');
    const lastUpdatedElement = document.getElementById('last-updated');
    const activeSymbolsElement = document.getElementById('active-symbols');
    const updateTimeElement = document.getElementById('update-time');
    const redisStatusElement = document.getElementById('redis-status');

    // Configuration display elements
    const listenerIdElement = document.getElementById('listener-id-config');
    const connectorTypeElement = document.getElementById('connector-type-config');
    const connectorCapacityElement = document.getElementById('connector-capacity-config');
    const selectedDataTypesElement = document.getElementById('selected-data-types');
    const listenerStatusElement = document.getElementById('listener-status-config');
    const exchangesConfigElement = document.getElementById('exchanges-config');

    let refreshInterval = 5000; // Default 5 seconds
    let refreshTimer = null;
    let eventSource = null;
    let configuredDataTypes = [];

    // Initialize the dashboard
    function initDashboard() {
        loadConfiguration();
        fetchData();
        setupEventSource();
        setupControls();
    }

    // Load configuration information
    function loadConfiguration() {
        fetch('/config_info')
            .then(response => response.json())
            .then(config => {
                updateConfigurationDisplay(config);
            })
            .catch(error => {
                console.error('Error loading configuration:', error);
                updateConfigurationDisplay({
                    error: 'Failed to load configuration',
                    listener_id: 'Unknown',
                    connector_type: 'Unknown',
                    connector_capacity: 'Unknown',
                    selected_data_types: [],
                    exchanges: [],
                    activated: false
                });
            });
    }

    // Update configuration display
    function updateConfigurationDisplay(config) {
        if (config.error) {
            listenerIdElement.textContent = config.listener_id;
            connectorTypeElement.textContent = config.connector_type;
            connectorCapacityElement.textContent = config.connector_capacity;
            selectedDataTypesElement.textContent = config.error;
            listenerStatusElement.textContent = 'Error';
            listenerStatusElement.className = 'badge bg-danger';
            exchangesConfigElement.innerHTML = `<div class="text-danger">${config.error}</div>`;
            return;
        }

        // Update basic info
        listenerIdElement.textContent = config.listener_id;
        
        // Add icon for connector type
        let connectorIcon = '';
        if (config.connector_type.includes('ccxt')) {
            connectorIcon = '<i class="bi bi-link-45deg"></i> ';
        } else if (config.connector_type.includes('oanda')) {
            connectorIcon = '<i class="bi bi-bank"></i> ';
        }
        connectorTypeElement.innerHTML = connectorIcon + config.connector_type;
        
        connectorCapacityElement.textContent = config.connector_capacity;
        
        // Update data types with badges
        const dataTypesBadges = config.selected_data_types.map(type => {
            let badgeClass = 'bg-secondary';
            let icon = '';
            if (type === 'candles') {
                badgeClass = 'bg-success';
                icon = '<i class="bi bi-graph-up"></i> ';
            } else if (type === 'trades') {
                badgeClass = 'bg-primary';
                icon = '<i class="bi bi-arrow-left-right"></i> ';
            } else if (type === 'ob') {
                badgeClass = 'bg-warning';
                icon = '<i class="bi bi-list-ol"></i> ';
            }
            
            return `<span class="badge ${badgeClass} me-1">${icon}${type}</span>`;
        }).join('');
        selectedDataTypesElement.innerHTML = dataTypesBadges;

        // Update status
        if (config.activated) {
            listenerStatusElement.textContent = 'Active';
            listenerStatusElement.className = 'badge bg-success';
        } else {
            listenerStatusElement.textContent = 'Inactive';
            listenerStatusElement.className = 'badge bg-secondary';
        }

        // Update exchanges and symbols
        if (config.exchanges && config.exchanges.length > 0) {
            let exchangesHtml = '';
            config.exchanges.forEach((exchange, index) => {
                const symbolsList = exchange.symbols.map(symbol => 
                    `<span class="badge bg-info me-1">${symbol}</span>`
                ).join('');
                
                exchangesHtml += `
                    <div class="mb-3 ${index > 0 ? 'mt-3' : ''} config-section">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <strong class="text-primary">
                                <i class="bi bi-currency-exchange"></i> ${exchange.name}
                            </strong>
                            <span class="badge bg-info">${exchange.symbols.length} symbols</span>
                        </div>
                        <div class="symbols-list p-2 bg-light rounded">
                            ${symbolsList}
                        </div>
                    </div>
                `;
            });
            exchangesConfigElement.innerHTML = exchangesHtml;
        } else {
            exchangesConfigElement.innerHTML = '<div class="text-muted">No exchanges configured</div>';
        }
    }

    // Fetch data from the API
    function fetchData() {
        fetch('/monitor_data')
            .then(response => response.json())
            .then(data => {
                updateDashboard(data);
                updateLastUpdated();
            })
            .catch(error => {
                console.error('Error fetching data:', error);
                setConnectionStatus(false);
            });
    }

    function formatTimestamp(timestamp) {
        // Create a Date object from the timestamp
        const date = new Date(timestamp);

        // Extract day, month, year, hours, minutes, and seconds
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0'); // Month is 0-based
        const year = date.getFullYear();
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');

        // Format the extracted components into dd/mm/yyyy hh:mm:ss
        return `${day}/${month}/${year} ${hours}:${minutes}:${seconds}`;
    }

    // Update the dashboard with new data
    function updateDashboard(data) {
        if (!data) return;

        const symbols = Object.keys(data).filter(key => key !== 'data_types');
        activeSymbolsElement.textContent = symbols.length;

        configuredDataTypes = data.data_types || [];

        let html = '';
        
        symbols.forEach(symbolKey => {
            const [exchange, symbol] = symbolKey.split(':');
            const symbolData = data[symbolKey];
            
            const showOrderbook = configuredDataTypes.includes('ob');
            const showTrades = configuredDataTypes.includes('trades');
            const showCandles = configuredDataTypes.includes('candles');
            
            // Determine which timestamp to display based on configured data types
            let displayTimestamp = null;
            if (showCandles && symbolData.candle?.timestamp) {
                displayTimestamp = symbolData.candle.timestamp;
            } else if (showTrades && symbolData.trades?.[0]?.timestamp) {
                displayTimestamp = symbolData.trades[0].timestamp;
            } else if (showOrderbook && symbolData.orderbook?.timestamp) {
                displayTimestamp = symbolData.orderbook.timestamp;
            }
            
            html += `
                <div class="col-md-6 col-lg-4 mb-4">
                    <div class="card symbol-card h-100">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">${symbol} <span class="badge bg-secondary">${exchange}</span></h5>
                            <span class="last-updated" data-symbol="${symbolKey}">${formatTimestamp(displayTimestamp) || 'N/A'}</span>
                        </div>
                        <div class="card-body">
                            ${!showOrderbook && !showTrades && !showCandles ? 
                                '<div class="text-center py-3 text-muted">No data types configured</div>' :
                                '<div class="row">' +
                                    (showOrderbook ? `
                                    <div class="col-md-6">
                                        <h6 class="card-subtitle mb-2 text-muted">Order Book</h6>
                                        <div class="d-flex justify-content-between">
                                            <div>
                                                <small class="text-muted">Best Bid</small>
                                                <p class="mb-1 positive" id="${symbolKey}-best-bid">${symbolData.orderbook?.bids?.[0]?.[0] || 'N/A'}</p>
                                            </div>
                                            <div>
                                                <small class="text-muted">Best Ask</small>
                                                <p class="mb-1 negative" id="${symbolKey}-best-ask">${symbolData.orderbook?.asks?.[0]?.[0] || 'N/A'}</p>
                                            </div>
                                        </div>
                                    </div>
                                    ` : '') +
                                    (showTrades ? `
                                    <div class="col-md-6">
                                        <h6 class="card-subtitle mb-2 text-muted">Last Trade</h6>
                                        <div class="d-flex justify-content-between">
                                            <div>
                                                <small class="text-muted">Price</small>
                                                <p class="mb-1 ${symbolData.trades?.[0]?.side === 'buy' ? 'positive' : 'negative'}" 
                                                   id="${symbolKey}-last-price">
                                                    ${symbolData.trades?.[0]?.price || 'N/A'}
                                                </p>
                                            </div>
                                            <div>
                                                <small class="text-muted">Side</small>
                                                <p class="mb-1" id="${symbolKey}-last-side">
                                                    ${symbolData.trades?.[0]?.side ? symbolData.trades[0].side.toUpperCase() : 'N/A'}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    ` : '') +
                                '</div>' +
                                (showCandles ? `
                                <div class="mt-3">
                                    <h6 class="card-subtitle mb-2 text-muted">Latest Candle (1m)</h6>
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <small class="text-muted">Open</small>
                                            <p class="mb-1" id="${symbolKey}-candle-open">${symbolData.candle?.open || 'N/A'}</p>
                                        </div>
                                        <div>
                                            <small class="text-muted">High</small>
                                            <p class="mb-1 positive" id="${symbolKey}-candle-high">${symbolData.candle?.high || 'N/A'}</p>
                                        </div>
                                        <div>
                                            <small class="text-muted">Low</small>
                                            <p class="mb-1 negative" id="${symbolKey}-candle-low">${symbolData.candle?.low || 'N/A'}</p>
                                        </div>
                                        <div>
                                            <small class="text-muted">Close</small>
                                            <p class="mb-1" id="${symbolKey}-candle-close">${symbolData.candle?.close || 'N/A'}</p>
                                        </div>
                                    </div>
                                </div>
                                ` : '')
                            }
                        </div>
                    </div>
                </div>
            `;
        });

        symbolsContainer.innerHTML = html || '<div class="col-12 text-center py-5"><p>No symbols being monitored</p></div>';
        setConnectionStatus(true);
    }

    // Set up Server-Sent Events for real-time updates
    function setupEventSource() {
        if (eventSource) eventSource.close();
        
        eventSource = new EventSource('/monitor_updates');
        
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
            updateLastUpdated();
        };
        
        eventSource.onerror = function() {
            setConnectionStatus(false);
            // Try to reconnect after 5 seconds
            setTimeout(setupEventSource, 5000);
        };
    }

    // Set up UI controls
    function setupControls() {
        // Set initial refresh interval
        refreshIntervalInput.value = refreshInterval / 1000;
        
        // Handle manual refresh
        refreshBtn.addEventListener('click', function() {
            clearTimeout(refreshTimer);
            fetchData();
            startRefreshTimer();
        });
        
        // Handle interval change
        refreshIntervalInput.addEventListener('change', function() {
            const seconds = parseInt(this.value);
            if (seconds >= 1) {
                refreshInterval = seconds * 1000;
                clearTimeout(refreshTimer);
                startRefreshTimer();
            }
        });
        
        startRefreshTimer();
    }

    // Start the refresh timer
    function startRefreshTimer() {
        refreshTimer = setTimeout(function() {
            fetchData();
            startRefreshTimer();
        }, refreshInterval);
    }

    // Update last updated timestamp
    function updateLastUpdated() {
        const now = new Date();
        lastUpdatedElement.textContent = `Last updated: ${now.toLocaleTimeString()}`;
        updateTimeElement.textContent = now.toLocaleTimeString();
        
        // Update individual symbol timestamps
        //document.querySelectorAll('.last-updated').forEach(el => {
        //    el.textContent = `Updated: ${now.toLocaleTimeString()}`;
        //});
    }

    // Set connection status indicator
    function setConnectionStatus(connected) {
        if (connected) {
            connectionStatus.className = 'badge bg-success';
            connectionStatus.innerHTML = '<span class="status-indicator status-active"></span> Connected';
        } else {
            connectionStatus.className = 'badge bg-danger';
            connectionStatus.innerHTML = '<span class="status-indicator status-inactive"></span> Disconnected';
        }
    }

    // Initialize Redis status check
    function checkRedisStatus() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                redisStatusElement.textContent = data.redis_status || 'Unknown';
            })
            .catch(error => {
                console.error('Error checking Redis status:', error);
                redisStatusElement.textContent = 'Error';
            });
    }

    // Initialize the dashboard
    initDashboard();
    checkRedisStatus();
    setInterval(checkRedisStatus, 30000); // Check Redis status every 30 seconds
});