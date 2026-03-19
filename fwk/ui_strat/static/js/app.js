let availableAssets = [];
let currentRunId = null;
let currentStrategy = 'dual_momentum';

function selectStrategy(strategy) {
    currentStrategy = strategy;
    document.getElementById('strategy-type').value = strategy;
    
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.strategy === strategy);
    });
    
    document.querySelectorAll('.strategy-params').forEach(panel => {
        panel.classList.remove('active');
    });
    
    if (strategy === 'dual_momentum') {
        document.getElementById('dual-momentum-params').classList.add('active');
    } else if (strategy === 'cto_line') {
        document.getElementById('cto-line-params').classList.add('active');
    }
}

async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        const data = await response.json();
        
        const select = document.getElementById('file-select');
        select.innerHTML = '<option value="">-- Select a file --</option>';
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        data.files.forEach(file => {
            const option = document.createElement('option');
            option.value = file.name;
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            option.textContent = `${file.name} (${sizeMB} MB)`;
            select.appendChild(option);
        });
    } catch (err) {
        showError('Failed to load files: ' + err.message);
    }
}

async function onFileSelect() {
    const filename = document.getElementById('file-select').value;
    const container = document.getElementById('assets-container');
    const fileInfo = document.getElementById('file-info');
    
    if (!filename) {
        container.innerHTML = '<p style="color: #888; text-align: center;">Select a file to see available assets</p>';
        fileInfo.textContent = '';
        document.getElementById('run-btn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`/api/files/${encodeURIComponent(filename)}/assets`);
        const data = await response.json();
        
        availableAssets = data.assets || [];
        fileInfo.textContent = `Found ${availableAssets.length} assets in file`;
        
        if (availableAssets.length === 0) {
            container.innerHTML = '<p style="color: #888; text-align: center;">No assets found in file</p>';
            return;
        }
        
        container.innerHTML = availableAssets.map((asset, idx) => `
            <div class="asset-item">
                <input type="checkbox" id="asset-${idx}" value="${asset}" checked>
                <label for="asset-${idx}">${asset}</label>
            </div>
        `).join('');
        
        populateDefaultAssetDropdown(availableAssets);
        updateRunButton();
    } catch (err) {
        showError('Failed to load assets: ' + err.message);
    }
}

function populateDefaultAssetDropdown(assets) {
    const selects = ['default-asset', 'cto-default-asset'];
    
    selects.forEach(selectId => {
        const select = document.getElementById(selectId);
        if (!select) return;
        select.innerHTML = '';
        
        assets.forEach(asset => {
            const option = document.createElement('option');
            option.value = asset;
            option.textContent = asset;
            select.appendChild(option);
        });
        
        if (assets.includes('SHV')) {
            select.value = 'SHV';
        } else if (assets.includes('TLT')) {
            select.value = 'TLT';
        } else if (assets.includes('GLD')) {
            select.value = 'GLD';
        } else if (assets.length > 0) {
            select.value = assets[0];
        }
    });
}

function selectAllAssets(selected) {
    const checkboxes = document.querySelectorAll('#assets-container input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = selected);
    updateRunButton();
}

function getSelectedAssets() {
    const checkboxes = document.querySelectorAll('#assets-container input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function updateRunButton() {
    const filename = document.getElementById('file-select').value;
    const selected = getSelectedAssets();
    document.getElementById('run-btn').disabled = !filename || selected.length === 0;
}

async function runBacktest() {
    hideError();
    
    const filename = document.getElementById('file-select').value;
    const selectedAssets = getSelectedAssets();
    const strategyType = document.getElementById('strategy-type').value;
    
    if (!filename || selectedAssets.length === 0) {
        showError('Please select a file and at least one asset');
        return;
    }
    
    document.getElementById('loading').classList.add('active');
    document.getElementById('results').classList.remove('active');
    document.getElementById('run-btn').disabled = true;
    
    try {
        const response = await fetch('/api/run-backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(buildRequestBody(filename, selectedAssets, strategyType))
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Backtest failed');
        }
        
        displayResults(data);
    } catch (err) {
        showError('Backtest failed: ' + err.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
        document.getElementById('run-btn').disabled = false;
    }
}

function buildRequestBody(filename, selectedAssets, strategyType) {
    const base = {
        filename: filename,
        selected_assets: selectedAssets,
        strategy_type: strategyType
    };
    
    if (strategyType === 'dual_momentum') {
        return {
            ...base,
            lookback: parseInt(document.getElementById('lookback').value),
            default_asset: document.getElementById('default-asset').value,
            top_n: parseInt(document.getElementById('top-n').value),
            abs_momentum_threshold: parseFloat(document.getElementById('abs-momentum-threshold').value),
            transaction_cost_pct: parseFloat(document.getElementById('transaction-cost').value),
            min_holding_periods: parseInt(document.getElementById('min-holding-periods').value),
            switch_threshold_pct: parseFloat(document.getElementById('switch-threshold').value),
            rsi_period: parseInt(document.getElementById('rsi-period').value),
            use_rsi_entry_filter: document.getElementById('use-rsi-entry-filter').checked,
            rsi_entry_max: parseFloat(document.getElementById('rsi-entry-max').value),
            use_rsi_entry_queue: document.getElementById('use-rsi-entry-queue').checked,
            use_rsi_diff_filter: document.getElementById('use-rsi-diff-filter').checked,
            rsi_diff_threshold: parseFloat(document.getElementById('rsi-diff-threshold').value)
        };
    } else if (strategyType === 'cto_line') {
        return {
            ...base,
            cto_params: [
                parseInt(document.getElementById('cto-v1').value),
                parseInt(document.getElementById('cto-m1').value),
                parseInt(document.getElementById('cto-m2').value),
                parseInt(document.getElementById('cto-v2').value)
            ],
            direction: document.getElementById('cto-direction').value,
            default_asset: document.getElementById('cto-default-asset').value || null,
            transaction_cost_pct: parseFloat(document.getElementById('cto-transaction-cost').value),
            min_holding_periods: parseInt(document.getElementById('cto-min-holding-periods').value),
            top_n: parseInt(document.getElementById('cto-top-n').value),
            abs_momentum_threshold: parseFloat(document.getElementById('cto-abs-momentum-threshold').value)
        };
    }
    return base;
}

function displayResults(data) {
    const metrics = data.metrics;
    const metricsHtml = `
        <div class="metric-card">
            <div class="metric-label">Period</div>
            <div class="metric-value">${metrics.years} years</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value ${metrics.total_return >= 0 ? 'positive' : 'negative'}">${metrics.total_return.toFixed(2)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CAGR</div>
            <div class="metric-value ${metrics.cagr >= 0 ? 'positive' : 'negative'}">${metrics.cagr.toFixed(2)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">${metrics.max_drawdown.toFixed(2)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">${metrics.sharpe_ratio.toFixed(2)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">${metrics.calmar_ratio.toFixed(2)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">${metrics.win_rate.toFixed(1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Orders</div>
            <div class="metric-value">${data.orders_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entries (Safe)</div>
            <div class="metric-value" style="color: #4ecca3; font-size: 1.1rem;">${data.entries_safe}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entries (Risky)</div>
            <div class="metric-value" style="color: #3498db; font-size: 1.1rem;">${data.entries_risky}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Exits (Safe)</div>
            <div class="metric-value" style="color: #f39c12; font-size: 1.1rem;">${data.exits_safe}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Exits (Risky)</div>
            <div class="metric-value" style="color: #e74c3c; font-size: 1.1rem;">${data.exits_risky}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Start Date</div>
            <div class="metric-value" style="font-size: 1rem;">${metrics.start_date}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">End Date</div>
            <div class="metric-value" style="font-size: 1rem;">${metrics.end_date}</div>
        </div>
    `;
    
    document.getElementById('metrics-grid').innerHTML = metricsHtml;
    
    let assetTableHtml = '';
    let nAssets = 0;
    let sumTotalReturn = 0, sumMaxDD = 0, sumSharpe = 0;
    let sumCandles = 0, sumPctAlloc = 0, sumStratReturn = 0, sumEntries = 0;
    
    if (data.asset_metrics) {
        for (const [asset, am] of Object.entries(data.asset_metrics)) {
            nAssets++;
            sumTotalReturn += am.total_return;
            sumMaxDD += am.max_drawdown;
            sumSharpe += am.sharpe_ratio;
            sumCandles += am.candles_allocated;
            sumPctAlloc += am.pct_allocated;
            sumStratReturn += am.strat_return;
            sumEntries += am.entries;
            
            assetTableHtml += `
                <tr>
                    <td>${asset}</td>
                    <td class="${am.total_return >= 0 ? 'positive' : 'negative'}">${am.total_return.toFixed(2)}%</td>
                    <td class="negative">${am.max_drawdown.toFixed(2)}%</td>
                    <td>${am.sharpe_ratio.toFixed(2)}</td>
                    <td>${am.candles_allocated.toLocaleString()}</td>
                    <td>${am.pct_allocated.toFixed(1)}%</td>
                    <td class="${am.strat_return >= 0 ? 'positive' : 'negative'}">${am.strat_return.toFixed(2)}%</td>
                    <td>${am.entries}</td>
                </tr>
            `;
        }
    }
    document.getElementById('asset-metrics-body').innerHTML = assetTableHtml;
    
    if (nAssets > 0) {
        const avgTotalReturn = sumTotalReturn / nAssets;
        const avgMaxDD = sumMaxDD / nAssets;
        const avgSharpe = sumSharpe / nAssets;
        
        const footerHtml = `
            <tr style="background: rgba(78, 204, 163, 0.1);">
                <td style="color: #4ecca3;">TOTAL/AVG</td>
                <td class="${avgTotalReturn >= 0 ? 'positive' : 'negative'}">${avgTotalReturn.toFixed(2)}%</td>
                <td class="negative">${avgMaxDD.toFixed(2)}%</td>
                <td>${avgSharpe.toFixed(2)}</td>
                <td>${sumCandles.toLocaleString()}</td>
                <td>${sumPctAlloc.toFixed(1)}%</td>
                <td class="${sumStratReturn >= 0 ? 'positive' : 'negative'}">${sumStratReturn.toFixed(2)}%</td>
                <td>${sumEntries}</td>
            </tr>
        `;
        document.getElementById('asset-metrics-footer').innerHTML = footerHtml;
    }
    
    const timestamp = Date.now();
    document.getElementById('chart-portfolio').src = `/static/charts/${data.charts.portfolio}?t=${timestamp}`;
    document.getElementById('chart-drawdown').src = `/static/charts/${data.charts.drawdown}?t=${timestamp}`;
    document.getElementById('chart-monthly').src = `/static/charts/${data.charts.monthly_returns}?t=${timestamp}`;
    document.getElementById('chart-sharpe').src = `/static/charts/${data.charts.sharpe}?t=${timestamp}`;
    
    if (data.current_positions && data.current_positions.length > 0) {
        let positionsHtml = '';
        data.current_positions.forEach(pos => {
            const pnlClass = pos.unrealized_pnl_pct >= 0 ? 'positive' : 'negative';
            positionsHtml += `
                <tr>
                    <td>${pos.asset}</td>
                    <td style="font-size: 0.85rem;">${pos.entry_date}</td>
                    <td>${pos.entry_price.toFixed(4)}</td>
                    <td>${pos.current_price.toFixed(4)}</td>
                    <td>${pos.quantity.toFixed(4)}</td>
                    <td>${pos.market_value.toFixed(2)}</td>
                    <td class="${pnlClass}">${pos.unrealized_pnl_pct.toFixed(2)}%</td>
                </tr>
            `;
        });
        document.getElementById('positions-body').innerHTML = positionsHtml;
        document.getElementById('positions-card').style.display = 'block';
    } else {
        document.getElementById('positions-card').style.display = 'none';
    }
    
    currentRunId = data.run_id;
    document.getElementById('trades-section').style.display = 'block';
    document.getElementById('trades-container').style.display = 'none';
    document.getElementById('trades-body').innerHTML = '';
    
    document.getElementById('allocations-section').style.display = 'block';
    document.getElementById('allocations-container').style.display = 'none';
    document.getElementById('allocations-body').innerHTML = '';
    
    document.getElementById('results').classList.add('active');
}

async function loadTrades() {
    if (!currentRunId) {
        showError('No backtest run available');
        return;
    }
    
    const btn = document.getElementById('show-trades-btn');
    const loading = document.getElementById('trades-loading');
    const container = document.getElementById('trades-container');
    
    btn.disabled = true;
    loading.style.display = 'block';
    container.style.display = 'none';
    
    try {
        const response = await fetch(`/api/run-backtest/${currentRunId}/trades`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load trades');
        }
        
        displayTrades(data.trades);
    } catch (err) {
        showError('Failed to load trades: ' + err.message);
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
        container.style.display = 'block';
    }
}

function displayTrades(trades) {
    document.getElementById('trades-count').textContent = trades.length;
    
    let tableHtml = '';
    trades.forEach((trade, idx) => {
        const returnClass = trade.return_pct >= 0 ? 'positive' : 'negative';
        tableHtml += `
            <tr>
                <td>${trade.asset}</td>
                <td style="font-size: 0.85rem;">${trade.entry_date}</td>
                <td>${trade.entry_price.toFixed(4)}</td>
                <td>${trade.entry_qty.toFixed(4)}</td>
                <td style="font-size: 0.85rem;">${trade.exit_date}</td>
                <td>${trade.exit_price.toFixed(4)}</td>
                <td>${trade.exit_qty.toFixed(4)}</td>
                <td class="${returnClass}">${trade.return_pct.toFixed(2)}%</td>
                <td class="negative">${trade.max_dd_pct.toFixed(2)}%</td>
                <td>${trade.duration_bars}</td>
            </tr>
        `;
    });
    
    document.getElementById('trades-body').innerHTML = tableHtml;
    document.getElementById('trades-container').style.display = 'block';
}

async function loadAllocations() {
    if (!currentRunId) {
        showError('No backtest run available');
        return;
    }
    
    const btn = document.getElementById('show-allocations-btn');
    const loading = document.getElementById('allocations-loading');
    const container = document.getElementById('allocations-container');
    
    btn.disabled = true;
    loading.style.display = 'block';
    container.style.display = 'none';
    
    try {
        const response = await fetch(`/api/run-backtest/${currentRunId}/allocations`);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load allocations');
        }
        
        displayAllocations(data.allocations);
    } catch (err) {
        showError('Failed to load allocations: ' + err.message);
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
        container.style.display = 'block';
    }
}

function displayAllocations(allocations) {
    if (!allocations || allocations.length === 0) {
        document.getElementById('allocations-body').innerHTML = '<tr><td colspan="100">No allocation data</td></tr>';
        document.getElementById('allocations-container').style.display = 'block';
        return;
    }
    
    document.getElementById('allocations-count').textContent = allocations.length;
    
    const firstRow = allocations[0];
    const assetKeys = Object.keys(firstRow).filter(k => k !== 'datetime' && k !== 'i_minute');
    
    let headerHtml = '<th>Datetime</th><th>i_minute</th>';
    assetKeys.forEach(asset => {
        headerHtml += `<th>${asset}</th>`;
    });
    document.getElementById('allocations-header').innerHTML = headerHtml;
    
    let tableHtml = '';
    allocations.forEach(row => {
        tableHtml += `<tr>`;
        tableHtml += `<td style="font-size: 0.85rem;">${row.datetime}</td>`;
        tableHtml += `<td>${row.i_minute}</td>`;
        assetKeys.forEach(asset => {
            const val = row[asset];
            const valClass = val > 0 ? 'positive' : '';
            tableHtml += `<td class="${valClass}">${val.toFixed(4)}</td>`;
        });
        tableHtml += `</tr>`;
    });
    
    document.getElementById('allocations-body').innerHTML = tableHtml;
    document.getElementById('allocations-container').style.display = 'block';
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.classList.add('active');
}

function hideError() {
    document.getElementById('error').classList.remove('active');
}

document.addEventListener('DOMContentLoaded', loadFiles);
document.getElementById('assets-container').addEventListener('change', updateRunButton);
