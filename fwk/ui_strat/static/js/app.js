let availableAssets = [];
let currentRunId = null;
let currentStrategy = 'dual_momentum';
let currentMetrics = null;
let currentParams = null;
let currentFilename = null;
let currentSelectedAssets = [];
let blendStrategies = [
    { type: 'dual_momentum', weight: 50, assets: [] },
    { type: 'cto_line', weight: 50, assets: [] }
];
let blendStrategyCounter = 2;

function safeToFixed(val, decimals = 2) {
    if (val === undefined || val === null || isNaN(val)) return '-';
    return Number(val).toFixed(decimals);
}

function selectStrategy(strategy) {
    currentStrategy = strategy;
    document.getElementById('strategy-type').value = strategy;
    
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.strategy === strategy);
    });
    
    document.querySelectorAll('.strategy-params').forEach(panel => {
        panel.classList.remove('active');
    });
    
    const runBtn = document.getElementById('run-btn');
    const runOptimBtn = document.getElementById('run-optim-btn');
    const runBlendBtn = document.getElementById('run-blend-btn');
    
    if (strategy === 'dual_momentum') {
        document.getElementById('dual-momentum-params').classList.add('active');
        runBtn.style.display = 'block';
        runOptimBtn.style.display = 'block';
        runBlendBtn.style.display = 'none';
    } else if (strategy === 'cto_line') {
        document.getElementById('cto-line-params').classList.add('active');
        runBtn.style.display = 'block';
        runOptimBtn.style.display = 'block';
        runBlendBtn.style.display = 'none';
    } else if (strategy === 'static_alloc') {
        document.getElementById('static-alloc-params').classList.add('active');
        runBtn.style.display = 'block';
        runOptimBtn.style.display = 'none';
        runBlendBtn.style.display = 'none';
        updateStaticAllocInputs();
    } else if (strategy === 'blend') {
        document.getElementById('blend-params').classList.add('active');
        runBtn.style.display = 'none';
        runOptimBtn.style.display = 'none';
        runBlendBtn.style.display = 'block';
        renderBlendAssetMatrix();
        validateBlendWeights();
    }
}

function getBlendStrategyParamsHTML(type, idx) {
    if (type === 'dual_momentum') {
        return `
            <div class="blend-param-row">
                <label>Lookback:</label>
                <input type="number" class="blend-param-lookback" value="3500" min="100" max="10000">
            </div>
            <div class="blend-param-row">
                <label>Default Asset:</label>
                <select class="blend-param-default-asset"></select>
            </div>
            <div class="blend-param-row">
                <label>Top N:</label>
                <input type="number" class="blend-param-top-n" value="2" min="1" max="10">
            </div>
            <div class="blend-param-row">
                <label>Abs Momentum Thresh:</label>
                <input type="number" class="blend-param-abs-momentum" value="0" step="0.01">
            </div>
            <div class="blend-param-row">
                <label>Min Holding:</label>
                <input type="number" class="blend-param-min-holding" value="240" min="0">
            </div>
            <div class="blend-param-row">
                <label>Switch Thresh %:</label>
                <input type="number" class="blend-param-switch-thresh" value="0" step="0.01">
            </div>
        `;
    } else if (type === 'cto_line') {
        return `
            <div class="blend-param-row">
                <label>V1/M1/M2/V2:</label>
                <input type="text" class="blend-param-cto-params" value="15,19,25,29" style="width: 120px;">
            </div>
            <div class="blend-param-row">
                <label>Direction:</label>
                <select class="blend-param-direction">
                    <option value="long">Long Only</option>
                    <option value="short">Short Only</option>
                    <option value="both" selected>Both</option>
                </select>
            </div>
            <div class="blend-param-row">
                <label>Default Asset:</label>
                <select class="blend-param-default-asset"></select>
            </div>
            <div class="blend-param-row">
                <label>Min Holding:</label>
                <input type="number" class="blend-param-min-holding" value="240" min="0">
            </div>
            <div class="blend-param-row">
                <label>Switch Thresh %:</label>
                <input type="number" class="blend-param-switch-thresh" value="0" step="0.01">
            </div>
        `;
    } else if (type === 'static_alloc') {
        return `
            <div class="blend-param-row">
                <label>Allocations (asset:pct,...):</label>
                <input type="text" class="blend-param-allocations" placeholder="QQQ:50,TLT:30,GLD:20" style="width: 150px;">
            </div>
            <div class="blend-param-row">
                <label>Rebalance Months:</label>
                <input type="number" class="blend-param-rebalance-months" value="0" min="0" max="120">
            </div>
        `;
    }
    return '';
}

function updateStaticAllocInputs() {
    const container = document.getElementById('static-allocations-container');
    const selectedAssets = getSelectedAssets();
    
    if (selectedAssets.length === 0) {
        container.innerHTML = '<p style="color: #888; text-align: center;">Select assets to configure allocations</p>';
        updateStaticAllocSum();
        return;
    }
    
    const defaultPct = (100 / selectedAssets.length).toFixed(1);
    
    let html = '<div class="static-alloc-grid">';
    selectedAssets.forEach(asset => {
        html += `
            <div class="static-alloc-row">
                <label>${asset}:</label>
                <input type="number" 
                       id="static-alloc-${asset}" 
                       class="static-alloc-input" 
                       data-asset="${asset}"
                       value="${defaultPct}" 
                       min="0" 
                       max="100" 
                       step="0.1"
                       onchange="updateStaticAllocSum()">
%
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
    updateStaticAllocSum();
}

function updateStaticAllocSum() {
    const inputs = document.querySelectorAll('.static-alloc-input');
    let total = 0;
    inputs.forEach(input => {
        total += parseFloat(input.value) || 0;
    });
    
    const totalSpan = document.getElementById('static-alloc-total');
    totalSpan.textContent = total.toFixed(1);
    
    const sumDiv = document.getElementById('static-alloc-sum');
    if (Math.abs(total - 100) < 0.1) {
        sumDiv.style.background = 'rgba(78, 204, 163, 0.2)';
        sumDiv.style.color = '#4ecca3';
    } else {
        sumDiv.style.background = 'rgba(231, 76, 60, 0.2)';
        sumDiv.style.color = '#e74c3c';
    }
}

function getStaticAllocations() {
    const inputs = document.querySelectorAll('.static-alloc-input');
    const allocations = {};
    inputs.forEach(input => {
        const asset = input.dataset.asset;
        const value = parseFloat(input.value) || 0;
        if (value > 0) {
            allocations[asset] = value;
        }
    });
    return allocations;
}

function onBlendStrategyTypeChange(idx) {
    const row = document.querySelector(`.blend-strategy-row[data-strategy-idx="${idx}"]`);
    if (!row) return;
    
    const typeSelect = row.querySelector('.blend-strategy-type');
    const type = typeSelect.value;
    
    blendStrategies[idx].type = type;
    
    const paramsDiv = row.querySelector('.blend-strategy-params');
    paramsDiv.innerHTML = getBlendStrategyParamsHTML(type, idx);
    
    populateBlendDefaultAssetDropdowns();
    renderBlendAssetMatrix();
    validateBlendWeights();
}

function addBlendStrategy() {
    const idx = blendStrategyCounter++;
    const container = document.getElementById('blend-strategies-container');
    
    const newRow = document.createElement('div');
    newRow.className = 'blend-strategy-row';
    newRow.dataset.strategyIdx = idx;
    newRow.innerHTML = `
        <div class="blend-strategy-header">
            <select class="blend-strategy-type" onchange="onBlendStrategyTypeChange(${idx})">
                <option value="dual_momentum">Dual Momentum</option>
                <option value="cto_line">CTO Line</option>
                <option value="static_alloc">Static Alloc</option>
            </select>
            <label class="blend-weight-label">Weight: <input type="number" class="blend-weight" value="0" min="0" max="100" step="1" onchange="validateBlendWeights()">%</label>
            <button class="btn-small blend-remove-btn" onclick="removeBlendStrategy(${idx})">Remove</button>
        </div>
        <div class="blend-strategy-params" id="blend-params-${idx}">
        </div>
    `;
    container.appendChild(newRow);
    
    blendStrategies.push({ type: 'dual_momentum', weight: 0, assets: [] });
    
    onBlendStrategyTypeChange(idx);
    updateBlendRemoveButtons();
    validateBlendWeights();
}

function removeBlendStrategy(idx) {
    const row = document.querySelector(`.blend-strategy-row[data-strategy-idx="${idx}"]`);
    if (row) {
        row.remove();
    }
    
    blendStrategies = blendStrategies.filter((s, i) => {
        const rowI = document.querySelector(`.blend-strategy-row[data-strategy-idx="${i}"]`);
        return rowI !== null;
    });
    
    updateBlendRemoveButtons();
    renderBlendAssetMatrix();
    validateBlendWeights();
}

function updateBlendRemoveButtons() {
    const rows = document.querySelectorAll('.blend-strategy-row');
    rows.forEach(row => {
        const btn = row.querySelector('.blend-remove-btn');
        btn.style.display = rows.length > 2 ? 'inline-block' : 'none';
    });
}

function validateBlendWeights() {
    const rows = document.querySelectorAll('.blend-strategy-row');
    let total = 0;
    let issues = [];
    
    if (!currentFilename) {
        issues.push('Select a data file');
    }
    
    rows.forEach((row, i) => {
        const weightInput = row.querySelector('.blend-weight');
        const weight = parseFloat(weightInput.value) || 0;
        total += weight;
        
        const idx = parseInt(row.dataset.strategyIdx);
        if (blendStrategies[idx] !== undefined) {
            blendStrategies[idx].weight = weight;
        }
    });
    
    if (Math.abs(total - 100) > 0.01) {
        issues.push(`Weights sum to ${total.toFixed(1)}% (need 100%)`);
    }
    
    rows.forEach((row, i) => {
        const idx = parseInt(row.dataset.strategyIdx);
        const checkboxes = document.querySelectorAll(`.blend-asset-checkbox[data-strategy-idx="${idx}"]:checked`);
        const typeSelect = row.querySelector('.blend-strategy-type');
        const type = typeSelect ? typeSelect.value : 'unknown';
        if (checkboxes.length === 0) {
            issues.push(`"${type === 'dual_momentum' ? 'Dual Momentum' : 'CTO Line'}" has no assets selected`);
        }
    });
    
    const statusEl = document.getElementById('blend-weight-status');
    const runBtn = document.getElementById('run-blend-btn');
    
    if (issues.length === 0) {
        statusEl.innerHTML = `<span style="color: #4ecca3;">Total: ${total.toFixed(1)}% ✓ - Ready to run</span>`;
        runBtn.disabled = false;
    } else {
        statusEl.innerHTML = `<span style="color: #e74c3c;">Missing: ${issues.join(' | ')}</span>`;
        runBtn.disabled = true;
    }
}

function populateBlendDefaultAssetDropdowns() {
    const rows = document.querySelectorAll('.blend-strategy-row');
    rows.forEach(row => {
        const select = row.querySelector('.blend-param-default-asset');
        if (!select) return;
        
        select.innerHTML = '';
        availableAssets.forEach(asset => {
            const option = document.createElement('option');
            option.value = asset;
            option.textContent = asset;
            select.appendChild(option);
        });
        
        if (availableAssets.includes('SHV')) {
            select.value = 'SHV';
        } else if (availableAssets.includes('TLT')) {
            select.value = 'TLT';
        } else if (availableAssets.length > 0) {
            select.value = availableAssets[0];
        }
    });
}

function renderBlendAssetMatrix() {
    const container = document.getElementById('blend-asset-matrix-container');
    
    if (availableAssets.length === 0) {
        container.innerHTML = '<p style="color: #888; text-align: center;">Select a file to see available assets</p>';
        return;
    }
    
    const rows = document.querySelectorAll('.blend-strategy-row');
    const strategyLabels = [];
    rows.forEach((row, i) => {
        const typeSelect = row.querySelector('.blend-strategy-type');
        const weightInput = row.querySelector('.blend-weight');
        const type = typeSelect ? typeSelect.value : 'dual_momentum';
        const weight = weightInput ? weightInput.value : '0';
        strategyLabels.push({ type, weight, idx: parseInt(row.dataset.strategyIdx) });
    });
    
    let html = '<table class="blend-asset-matrix-table"><thead><tr><th>Asset</th>';
    strategyLabels.forEach((s, i) => {
        html += `<th>${s.type === 'dual_momentum' ? 'Dual Mom' : 'CTO Line'}<br>(${s.weight}%)</th>`;
    });
    html += '</tr></thead><tbody>';
    
    availableAssets.forEach(asset => {
        html += `<tr><td>${asset}</td>`;
        strategyLabels.forEach((s, i) => {
            const isChecked = blendStrategies[s.idx] && blendStrategies[s.idx].assets.includes(asset);
            html += `<td><input type="checkbox" class="blend-asset-checkbox" data-asset="${asset}" data-strategy-idx="${s.idx}" ${isChecked ? 'checked' : ''} onchange="onBlendAssetChange()"></td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    html += `<div style="margin-top: 10px;">
        <button class="select-all-btn" onclick="selectAllBlendAssets(true)">Select All for All</button>
        <button class="select-all-btn" onclick="selectAllBlendAssets(false)">Deselect All</button>
    </div>`;
    
    container.innerHTML = html;
    
    validateBlendWeights();
}

function onBlendAssetChange() {
    const checkboxes = document.querySelectorAll('.blend-asset-checkbox');
    
    checkboxes.forEach(cb => {
        const asset = cb.dataset.asset;
        const idx = parseInt(cb.dataset.strategyIdx);
        
        if (!blendStrategies[idx]) {
            blendStrategies[idx] = { type: 'dual_momentum', weight: 0, assets: [] };
        }
        
        if (cb.checked) {
            if (!blendStrategies[idx].assets.includes(asset)) {
                blendStrategies[idx].assets.push(asset);
            }
        } else {
            blendStrategies[idx].assets = blendStrategies[idx].assets.filter(a => a !== asset);
        }
    });
    
    validateBlendWeights();
}

function selectAllBlendAssets(selected) {
    const checkboxes = document.querySelectorAll('.blend-asset-checkbox');
    checkboxes.forEach(cb => cb.checked = selected);
    onBlendAssetChange();
}

function buildBlendRequestBody() {
    const rows = document.querySelectorAll('.blend-strategy-row');
    const strategies = [];
    
    rows.forEach(row => {
        const idx = parseInt(row.dataset.strategyIdx);
        const typeSelect = row.querySelector('.blend-strategy-type');
        const weightInput = row.querySelector('.blend-weight');
        
        const type = typeSelect.value;
        const weight = parseFloat(weightInput.value) || 0;
        
        let assets = [];
        const checkboxes = document.querySelectorAll(`.blend-asset-checkbox[data-strategy-idx="${idx}"]:checked`);
        checkboxes.forEach(cb => assets.push(cb.dataset.asset));
        
        const params = {};
        
        if (type === 'dual_momentum') {
            params.lookback = parseInt(row.querySelector('.blend-param-lookback')?.value || 3500);
            params.default_asset = row.querySelector('.blend-param-default-asset')?.value;
            params.top_n = parseInt(row.querySelector('.blend-param-top-n')?.value || 2);
            params.abs_momentum_threshold = parseFloat(row.querySelector('.blend-param-abs-momentum')?.value || 0);
            params.min_holding_periods = parseInt(row.querySelector('.blend-param-min-holding')?.value || 240);
            params.switch_threshold_pct = parseFloat(row.querySelector('.blend-param-switch-thresh')?.value || 0);
        } else if (type === 'cto_line') {
            const ctoParamsStr = row.querySelector('.blend-param-cto-params')?.value || '15,19,25,29';
            params.cto_params = ctoParamsStr.split(',').map(s => parseInt(s.trim()));
            params.direction = row.querySelector('.blend-param-direction')?.value || 'both';
            params.default_asset = row.querySelector('.blend-param-default-asset')?.value;
            params.min_holding_periods = parseInt(row.querySelector('.blend-param-min-holding')?.value || 240);
            params.switch_threshold_pct = parseFloat(row.querySelector('.blend-param-switch-thresh')?.value || 0);
        } else if (type === 'static_alloc') {
            const allocStr = row.querySelector('.blend-param-allocations')?.value || '';
            const allocations = {};
            if (allocStr) {
                allocStr.split(',').forEach(pair => {
                    const [asset, pct] = pair.split(':').map(s => s.trim());
                    if (asset && pct) {
                        allocations[asset] = parseFloat(pct);
                    }
                });
            }
            params.allocations = allocations;
            params.rebalance_months = parseInt(row.querySelector('.blend-param-rebalance-months')?.value || 0);
        }
        
        strategies.push({
            strategy_type: type,
            weight: weight,
            assets: assets,
            params: params
        });
    });
    
    return {
        filename: currentFilename,
        strategies: strategies,
        transaction_cost_pct: parseFloat(document.getElementById('blend-transaction-cost').value)
    };
}

async function runBlendBacktest() {
    hideError();
    
    if (!currentFilename) {
        showError('Please select a file first');
        return;
    }
    
    const requestBody = buildBlendRequestBody();
    
    for (const s of requestBody.strategies) {
        if (s.assets.length === 0) {
            showError(`Strategy "${s.strategy_type}" has no assets selected`);
            return;
        }
    }
    
    currentSelectedAssets = [...new Set(requestBody.strategies.flatMap(s => s.assets))];
    currentParams = { strategies: requestBody.strategies };
    
    document.getElementById('loading').classList.add('active');
    document.getElementById('results').classList.remove('active');
    document.getElementById('run-blend-btn').disabled = true;
    
    try {
        const response = await fetch('/api/run-blend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Blend backtest failed');
        }
        
        currentMetrics = data.metrics;
        currentMetrics.orders_count = data.orders_count;
        
        displayResults(data);
    } catch (err) {
        showError('Blend backtest failed: ' + err.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
        document.getElementById('run-blend-btn').disabled = false;
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
    
    currentFilename = filename || null;
    
    if (!filename) {
        container.innerHTML = '<p style="color: #888; text-align: center;">Select a file to see available assets</p>';
        fileInfo.textContent = '';
        document.getElementById('run-btn').disabled = true;
        currentFilename = null;
        validateBlendWeights();
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
                <input type="checkbox" id="asset-${idx}" value="${asset}" checked onchange="onAssetCheckboxChange()">
                <label for="asset-${idx}">${asset}</label>
            </div>
        `).join('');
        
        currentFilename = filename;
        
        populateDefaultAssetDropdown(availableAssets);
        populateBlendDefaultAssetDropdowns();
        renderBlendAssetMatrix();
        updateStaticAllocInputs();
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
    onAssetCheckboxChange();
}

function onAssetCheckboxChange() {
    updateRunButton();
    updateStaticAllocInputs();
}

function getSelectedAssets() {
    const checkboxes = document.querySelectorAll('#assets-container input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function updateRunButton() {
    const filename = document.getElementById('file-select').value;
    const selected = getSelectedAssets();
    const disabled = !filename || selected.length === 0;
    document.getElementById('run-btn').disabled = disabled;
    document.getElementById('run-optim-btn').disabled = disabled;
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
    
    currentFilename = filename;
    currentSelectedAssets = selectedAssets;
    
    document.getElementById('loading').classList.add('active');
    document.getElementById('results').classList.remove('active');
    document.getElementById('run-btn').disabled = true;
    document.getElementById('run-optim-btn').disabled = true;
    
    try {
        const requestBody = buildRequestBody(filename, selectedAssets, strategyType);
        currentParams = extractParams(requestBody);
        
        const response = await fetch('/api/run-backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Backtest failed');
        }
        
        currentMetrics = data.metrics;
        currentMetrics.orders_count = data.orders_count;
        
        displayResults(data);
    } catch (err) {
        showError('Backtest failed: ' + err.message);
    } finally {
        document.getElementById('loading').classList.remove('active');
        document.getElementById('run-btn').disabled = false;
        document.getElementById('run-optim-btn').disabled = false;
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
    } else if (strategyType === 'static_alloc') {
        return {
            ...base,
            allocations: getStaticAllocations(),
            rebalance_months: parseInt(document.getElementById('static-rebalance-months').value),
            transaction_cost_pct: parseFloat(document.getElementById('static-transaction-cost').value)
        };
    }
    return base;
}

function displayResults(data) {
    const metrics = data.metrics;
    const metricsHtml = `
        <div class="metric-card">
            <div class="metric-label">Period</div>
            <div class="metric-value">${safeToFixed(metrics.years, 2)} years</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value ${metrics.total_return >= 0 ? 'positive' : 'negative'}">${safeToFixed(metrics.total_return)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CAGR</div>
            <div class="metric-value ${metrics.cagr >= 0 ? 'positive' : 'negative'}">${safeToFixed(metrics.cagr)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">${safeToFixed(metrics.max_drawdown)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">${safeToFixed(metrics.sharpe_ratio)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">${safeToFixed(metrics.profit_factor)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Longest Underwater</div>
            <div class="metric-value">${metrics.longest_underwater_days ?? '-'} days</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value">${safeToFixed(metrics.calmar_ratio)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">${safeToFixed(metrics.win_rate, 1)}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Orders</div>
            <div class="metric-value">${data.orders_count ?? 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entries (Safe)</div>
            <div class="metric-value" style="color: #4ecca3; font-size: 1.1rem;">${data.entries_safe ?? 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entries (Risky)</div>
            <div class="metric-value" style="color: #3498db; font-size: 1.1rem;">${data.entries_risky ?? 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Exits (Safe)</div>
            <div class="metric-value" style="color: #f39c12; font-size: 1.1rem;">${data.exits_safe ?? 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Exits (Risky)</div>
            <div class="metric-value" style="color: #e74c3c; font-size: 1.1rem;">${data.exits_risky ?? 0}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Start Date</div>
            <div class="metric-value" style="font-size: 1rem;">${metrics.start_date ?? '-'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">End Date</div>
            <div class="metric-value" style="font-size: 1rem;">${metrics.end_date ?? '-'}</div>
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
            sumTotalReturn += am.total_return || 0;
            sumMaxDD += am.max_drawdown || 0;
            sumSharpe += am.sharpe_ratio || 0;
            sumCandles += am.candles_allocated || 0;
            sumPctAlloc += am.pct_allocated || 0;
            sumStratReturn += am.strat_return || 0;
            sumEntries += am.entries || 0;
            
            assetTableHtml += `
                <tr>
                    <td>${asset}</td>
                    <td class="${am.total_return >= 0 ? 'positive' : 'negative'}">${safeToFixed(am.total_return)}%</td>
                    <td class="negative">${safeToFixed(am.max_drawdown)}%</td>
                    <td>${safeToFixed(am.sharpe_ratio)}</td>
                    <td>${(am.candles_allocated || 0).toLocaleString()}</td>
                    <td>${safeToFixed(am.pct_allocated, 1)}%</td>
                    <td class="${am.strat_return >= 0 ? 'positive' : 'negative'}">${safeToFixed(am.strat_return)}%</td>
                    <td>${am.entries || 0}</td>
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
                <td class="${avgTotalReturn >= 0 ? 'positive' : 'negative'}">${safeToFixed(avgTotalReturn)}%</td>
                <td class="negative">${safeToFixed(avgMaxDD)}%</td>
                <td>${safeToFixed(avgSharpe)}</td>
                <td>${sumCandles.toLocaleString()}</td>
                <td>${safeToFixed(sumPctAlloc, 1)}%</td>
                <td class="${sumStratReturn >= 0 ? 'positive' : 'negative'}">${safeToFixed(sumStratReturn)}%</td>
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
            const sideClass = pos.side === 'long' ? 'positive' : (pos.side === 'short' ? 'negative' : '');
            const sideDisplay = pos.side ? pos.side.toUpperCase() : '-';
            positionsHtml += `
                <tr>
                    <td>${pos.asset}</td>
                    <td class="${sideClass}" style="font-weight: 600;">${sideDisplay}</td>
                    <td style="font-size: 0.85rem;">${pos.entry_date ?? '-'}</td>
                    <td>${safeToFixed(pos.entry_price, 4)}</td>
                    <td>${safeToFixed(pos.current_price, 4)}</td>
                    <td>${safeToFixed(pos.quantity, 4)}</td>
                    <td>${safeToFixed(pos.market_value)}</td>
                    <td class="${pnlClass}">${safeToFixed(pos.unrealized_pnl_pct)}%</td>
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
        const sideClass = trade.side === 'long' ? 'positive' : (trade.side === 'short' ? 'negative' : '');
        const sideDisplay = trade.side ? trade.side.toUpperCase() : '-';
        tableHtml += `
            <tr>
                <td>${trade.asset}</td>
                <td class="${sideClass}" style="font-weight: 600;">${sideDisplay}</td>
                <td style="font-size: 0.85rem;">${trade.entry_date ?? '-'}</td>
                <td>${safeToFixed(trade.entry_price, 4)}</td>
                <td>${safeToFixed(trade.entry_qty, 4)}</td>
                <td style="font-size: 0.85rem;">${trade.exit_date ?? '-'}</td>
                <td>${safeToFixed(trade.exit_price, 4)}</td>
                <td>${safeToFixed(trade.exit_qty, 4)}</td>
                <td class="${returnClass}">${safeToFixed(trade.return_pct)}%</td>
                <td class="negative">${safeToFixed(trade.max_dd_pct)}%</td>
                <td>${trade.duration_bars ?? 0}</td>
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
        tableHtml += `<td style="font-size: 0.85rem;">${row.datetime ?? '-'}</td>`;
        tableHtml += `<td>${row.i_minute ?? 0}</td>`;
        assetKeys.forEach(asset => {
            const val = row[asset];
            const valClass = val > 0 ? 'positive' : '';
            tableHtml += `<td class="${valClass}">${safeToFixed(val, 4)}</td>`;
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

function extractParams(requestBody) {
    const params = {};
    const paramKeys = ['lookback', 'default_asset', 'top_n', 'abs_momentum_threshold', 
                       'transaction_cost_pct', 'min_holding_periods', 'switch_threshold_pct',
                       'rsi_period', 'use_rsi_entry_filter', 'rsi_entry_max',
                       'use_rsi_entry_queue', 'use_rsi_diff_filter', 'rsi_diff_threshold',
                       'cto_params', 'direction', 'cap_to_half_assets'];
    paramKeys.forEach(key => {
        if (requestBody[key] !== undefined) {
            params[key] = requestBody[key];
        }
    });
    return params;
}

async function runOptimization() {
    hideError();
    
    const filename = document.getElementById('file-select').value;
    const selectedAssets = getSelectedAssets();
    
    if (!filename || selectedAssets.length === 0) {
        showError('Please select a file and at least one asset');
        return;
    }
    
    const defaultAsset = document.getElementById('default-asset').value;
    if (!defaultAsset) {
        showError('Please select a default asset');
        return;
    }
    
    document.getElementById('optim-loading').style.display = 'block';
    document.getElementById('optim-results').style.display = 'none';
    document.getElementById('run-optim-btn').disabled = true;
    document.getElementById('run-btn').disabled = true;
    
    try {
        const response = await fetch('/api/run-optim', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename,
                selected_assets: selectedAssets,
                default_asset: defaultAsset,
                top_n: parseInt(document.getElementById('top-n').value),
                abs_momentum_threshold: parseFloat(document.getElementById('abs-momentum-threshold').value),
                transaction_cost_pct: parseFloat(document.getElementById('transaction-cost').value),
                min_holding_periods: parseInt(document.getElementById('min-holding-periods').value),
                switch_threshold_pct: parseFloat(document.getElementById('switch-threshold').value),
                lookback_min: 100,
                lookback_max: 10000,
                num_steps: 50
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Optimization failed');
        }
        
        displayOptimResults(data);
    } catch (err) {
        showError('Optimization failed: ' + err.message);
    } finally {
        document.getElementById('optim-loading').style.display = 'none';
        document.getElementById('run-optim-btn').disabled = false;
        document.getElementById('run-btn').disabled = false;
    }
}

function displayOptimResults(data) {
    const optimSection = document.getElementById('optim-results');
    const summaryDiv = document.getElementById('optim-summary');
    
    let summaryHtml = '';
    
    if (data.best_by_cagr) {
        summaryHtml += `
            <div class="metric-card">
                <div class="metric-label">Best CAGR</div>
                <div class="metric-value positive">${data.best_by_cagr.cagr}%</div>
                <div style="font-size: 0.8rem; color: #888;">Lookback: ${data.best_by_cagr.lookback}</div>
            </div>
        `;
    }
    
    if (data.best_by_sharpe) {
        summaryHtml += `
            <div class="metric-card">
                <div class="metric-label">Best Sharpe-like</div>
                <div class="metric-value">${data.best_by_sharpe.cagr}%</div>
                <div style="font-size: 0.8rem; color: #888;">Lookback: ${data.best_by_sharpe.lookback}</div>
            </div>
        `;
    }
    
    summaryHtml += `
        <div class="metric-card">
            <div class="metric-label">Lookback Range</div>
            <div class="metric-value" style="font-size: 1rem;">${data.lookback_range[0]} - ${data.lookback_range[1]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Points Tested</div>
            <div class="metric-value">${data.num_steps}</div>
        </div>
    `;
    
    summaryDiv.innerHTML = summaryHtml;
    
    if (data.chart) {
        const timestamp = Date.now();
        document.getElementById('chart-optim').src = `/static/charts/${data.chart}?t=${timestamp}`;
    }
    
    optimSection.style.display = 'block';
}

async function saveExperiment() {
    if (!currentMetrics || !currentParams) {
        showError('No backtest results to save. Run a backtest first.');
        return;
    }
    
    const name = document.getElementById('exp-name').value.trim() || null;
    
    try {
        const response = await fetch('/api/save-experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                filename: currentFilename,
                strategy_type: currentStrategy,
                params: currentParams,
                metrics: currentMetrics,
                selected_assets: currentSelectedAssets
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to save experiment');
        }
        
        document.getElementById('exp-name').value = '';
        alert('Experiment saved successfully! ID: ' + data.id);
        loadExperiments();
    } catch (err) {
        showError('Failed to save experiment: ' + err.message);
    }
}

async function loadExperiments() {
    const container = document.getElementById('experiments-container');
    const loading = document.getElementById('experiments-loading');
    
    loading.style.display = 'block';
    container.innerHTML = '';
    
    try {
        const response = await fetch('/api/experiments');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load experiments');
        }
        
        displayExperiments(data.experiments);
    } catch (err) {
        container.innerHTML = `<p style="color: #e74c3c; text-align: center;">Error: ${err.message}</p>`;
    } finally {
        loading.style.display = 'none';
    }
}

function displayExperiments(experiments) {
    const container = document.getElementById('experiments-container');
    
    if (!experiments || experiments.length === 0) {
        container.innerHTML = '<p style="color: #888; text-align: center;">No experiments saved yet.</p>';
        return;
    }
    
    let tableHtml = `
        <div style="max-height: 400px; overflow-y: auto;">
            <table class="asset-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Name</th>
                        <th>File</th>
                        <th>Strategy</th>
                        <th>CAGR</th>
                        <th>Max DD</th>
                        <th>Lookback</th>
                        <th>Top N</th>
                        <th>Date</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    experiments.forEach(exp => {
        const cagrClass = (exp.cagr || 0) >= 0 ? 'positive' : 'negative';
        const displayName = exp.name || '-';
        const shortFile = exp.filename.length > 15 ? exp.filename.substring(0, 15) + '...' : exp.filename;
        
        tableHtml += `
            <tr>
                <td>${exp.id}</td>
                <td style="font-size: 0.8rem; max-width: 120px; overflow: hidden; text-overflow: ellipsis;" title="${exp.name || ''}">${displayName}</td>
                <td style="font-size: 0.8rem;" title="${exp.filename}">${shortFile}</td>
                <td>${exp.strategy_type}</td>
                <td class="${cagrClass}">${exp.cagr ? exp.cagr.toFixed(2) + '%' : '-'}</td>
                <td class="negative">${exp.max_dd ? exp.max_dd.toFixed(2) + '%' : '-'}</td>
                <td>${exp.lookback || '-'}</td>
                <td>${exp.top_n || '-'}</td>
                <td style="font-size: 0.75rem;">${exp.created_at ? exp.created_at.substring(0, 10) : '-'}</td>
                <td>
                    <button class="btn-small" onclick="deleteExperiment(${exp.id})" style="padding: 4px 8px; font-size: 0.75rem;">Delete</button>
                </td>
            </tr>
        `;
    });
    
    tableHtml += '</tbody></table></div>';
    container.innerHTML = tableHtml;
}

async function deleteExperiment(expId) {
    if (!confirm('Are you sure you want to delete experiment #' + expId + '?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/experiments/${expId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to delete experiment');
        }
        
        loadExperiments();
    } catch (err) {
        showError('Failed to delete experiment: ' + err.message);
    }
}

document.addEventListener('DOMContentLoaded', loadFiles);
document.getElementById('assets-container').addEventListener('change', updateRunButton);
