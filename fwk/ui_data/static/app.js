let currentTaskId = null;
let selectedAssets = [];
let statusCheckInterval = null;

const freqSelect = document.getElementById('freqSelect');
const productTypeSelect = document.getElementById('productTypeSelect');
const loadAssetsBtn = document.getElementById('loadAssetsBtn');
const assetsPanel = document.getElementById('assetsPanel');
const assetsGrid = document.getElementById('assetsGrid');
const assetSearch = document.getElementById('assetSearch');
const selectAllBtn = document.getElementById('selectAllBtn');
const deselectAllBtn = document.getElementById('deselectAllBtn');
const selectedCount = document.getElementById('selectedCount');
const optionsPanel = document.getElementById('optionsPanel');
const computeFeatures = document.getElementById('computeFeatures');
const outputPrefix = document.getElementById('outputPrefix');
const createBundleBtn = document.getElementById('createBundleBtn');
const progressPanel = document.getElementById('progressPanel');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultPanel = document.getElementById('resultPanel');
const resultContent = document.getElementById('resultContent');
const bundlesList = document.getElementById('bundlesList');
const selectedAssetsContainer = document.getElementById('selectedAssetsContainer');

const featureConfigSelect = document.getElementById('featureConfigSelect');

let currentTaskId = null;
let selectedAssets = [];
let statusCheckInterval = null;
let featureConfigs = [];

async function loadAssets() {
    const freq = freqSelect.value;
    const productType = productTypeSelect.value;
    
    assetsGrid.innerHTML = '<div class="loading">Loading assets...</div>';
    
    try {
        const url = `/api/assets/list?freq=${encodeURIComponent(freq)}${productType ? '&product_type=' + encodeURIComponent(productType) : ''}`;
        const response = await fetch(url);
        const data = await response.json();
        
        renderAssets(data.assets_by_type);
        assetsPanel.style.display = 'block';
        optionsPanel.style.display = 'block';
    } catch (error) {
        assetsGrid.innerHTML = `<div class="empty-state">Error loading assets: ${error.message}</div>`;
    }
}

function renderAssets(assetsByType) {
    if (!assetsByType || Object.keys(assetsByType).length === 0) {
        assetsGrid.innerHTML = '<div class="empty-state">No assets found for selected filters</div>';
        return;
    }
    
    selectedAssets = [];
    let html = '';
    
    for (const [productType, assets] of Object.entries(assetsByType)) {
        for (const asset of assets) {
            html += `
                <div class="asset-item" data-symbol="${asset.symbol}" data-product-type="${asset.product_type}">
                    <input type="checkbox" id="asset_${asset.symbol}" onchange="updateSelection()">
                    <label for="asset_${asset.symbol}">${asset.symbol}</label>
                    <span class="asset-type-label">${asset.product_type}</span>
                </div>
            `;
        }
    }
    
    assetsGrid.innerHTML = html;
    updateSelectedCount();
}

window.updateSelection = function() {
    const checkboxes = assetsGrid.querySelectorAll('input[type="checkbox"]:checked');
    selectedAssets = [];
    
    checkboxes.forEach(cb => {
        const item = cb.closest('.asset-item');
        selectedAssets.push({
            symbol: item.dataset.symbol,
            product_type: item.dataset.productType
        });
    });
    
    updateSelectedCount();
};

function updateSelectedCount() {
    selectedCount.textContent = `${selectedAssets.length} assets selected`;
    createBundleBtn.disabled = selectedAssets.length === 0;
    renderSelectedChips();
}

function renderSelectedChips() {
    if (selectedAssets.length === 0) {
        selectedAssetsContainer.innerHTML = '';
        return;
    }
    
    let html = '';
    selectedAssets.forEach(asset => {
        html += `
            <span class="selected-chip">
                ${asset.symbol}
                <span class="remove-btn" onclick="removeAsset('${asset.symbol}')">&times;</span>
            </span>
        `;
    });
    selectedAssetsContainer.innerHTML = html;
}

window.removeAsset = function(symbol) {
    const checkbox = document.getElementById(`asset_${symbol}`);
    if (checkbox) {
        checkbox.checked = false;
        updateSelection();
    }
};

loadAssetsBtn.addEventListener('click', loadAssets);

assetSearch.addEventListener('input', (e) => {
    const search = e.target.value.toLowerCase();
    const items = assetsGrid.querySelectorAll('.asset-item');
    items.forEach(item => {
        const symbol = item.dataset.symbol.toLowerCase();
        item.style.display = symbol.includes(search) ? 'flex' : 'none';
    });
});

selectAllBtn.addEventListener('click', () => {
    const visibleCheckboxes = assetsGrid.querySelectorAll('.asset-item:not([style*="display: none"]) input[type="checkbox"]');
    visibleCheckboxes.forEach(cb => cb.checked = true);
    updateSelection();
});

deselectAllBtn.addEventListener('click', () => {
    const checkboxes = assetsGrid.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => cb.checked = false);
    updateSelection();
});

createBundleBtn.addEventListener('click', async () => {
    if (selectedAssets.length === 0) return;
    
    createBundleBtn.disabled = true;
    progressPanel.style.display = 'block';
    resultPanel.style.display = 'none';
    
    const assets = selectedAssets.map(a => a.symbol);
    const productTypes = selectedAssets.map(a => a.product_type);
    const selectedConfig = featureConfigSelect.value || null;
    
    try {
        const response = await fetch('/api/bundle/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                assets: assets,
                product_types: productTypes,
                freq: freqSelect.value,
                output_prefix: outputPrefix.value || 'custom_bundle',
                compute_features: computeFeatures.checked,
                feature_config_path: selectedConfig,
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentTaskId = data.task_id;
            startStatusCheck();
        } else {
            showResult(false, data.detail || 'Failed to create bundle')
        }
    } catch (error) {
        showResult(false, error.message);
    }
});
        
        const data = await response.json();
        
        if (response.ok) {
            currentTaskId = data.task_id;
            startStatusCheck();
        } else {
            showResult(false, data.detail || 'Failed to create bundle');
        }
    } catch (error) {
        showResult(false, error.message);
    }
});

function startStatusCheck() {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/bundle/${currentTaskId}/status`);
            const status = await response.json();
            
            const percent = status.total > 0 ? Math.round((status.progress / status.total) * 100) : 0;
            progressFill.style.width = `${percent}%`;
            progressText.textContent = status.message || `Processing ${status.progress}/${status.total} assets...`;
            
            if (status.status === 'completed') {
                clearInterval(statusCheckInterval);
                await showBundleResult();
            } else if (status.status === 'failed') {
                clearInterval(statusCheckInterval);
                showResult(false, status.error || 'Bundle creation failed');
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }, 1000);
}

async function showBundleResult() {
    try {
        const response = await fetch(`/api/bundle/${currentTaskId}/result`);
        const result = await response.json();
        
        if (result.success) {
            showResult(true, `
                <strong>Bundle created successfully!</strong><br>
                <br>
                <strong>Path:</strong> ${result.output_path}<br>
                <strong>Size:</strong> ${result.file_size_mb} MB<br>
                <strong>Rows:</strong> ${result.total_rows?.toLocaleString()}<br>
                <strong>Features:</strong> ${result.feature_count}<br>
                <strong>Assets:</strong> ${result.assets_processed}
            `);
        } else {
            showResult(false, result.error || 'Unknown error');
        }
    } catch (error) {
        showResult(false, error.message);
    }
}

function showResult(success, message) {
    progressPanel.style.display = 'none';
    resultPanel.style.display = 'block';
    resultContent.innerHTML = `<div class="${success ? 'result-success' : 'result-error'}">${message}</div>`;
    createBundleBtn.disabled = false;
    loadBundles();
}

async function loadBundles() {
    try {
        const response = await fetch('/api/bundle/list');
        const bundles = await response.json();
        
        if (bundles.length === 0) {
            bundlesList.innerHTML = '<div class="empty-state">No bundles found</div>';
            return;
        }
        
        let html = '';
        bundles.forEach(bundle => {
            html += `
                <div class="bundle-item">
                    <div>
                        <div class="bundle-name">${bundle.filename}</div>
                        <div class="bundle-info">${bundle.size_mb} MB | Modified: ${new Date(bundle.modified).toLocaleString()}</div>
                    </div>
                </div>
            `;
        });
        bundlesList.innerHTML = html;
    } catch (error) {
        bundlesList.innerHTML = `<div class="empty-state">Error loading bundles: ${error.message}</div>`;
    }
}

loadBundles();
