let currentTaskId = null;
let selectedAssets = [];
let statusCheckInterval = null;
let featureConfigs = [];

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
const bundleModal = document.getElementById('bundleModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');
const folderSelect = document.getElementById('folderSelect');
const newFolderInput = document.getElementById('newFolderInput');
const filenamePreview = document.getElementById('filenamePreview');

function getFreqSuffix(freq) {
    if (freq.includes('1hour') || freq.includes('1_hour')) return '_1hour';
    if (freq.includes('1min') || freq.includes('1_min')) return '_1min';
    if (freq.includes('5min') || freq.includes('5_min')) return '_5min';
    if (freq.includes('15min') || freq.includes('15_min')) return '_15min';
    return `_${freq}`;
}

function updateFilenamePreview() {
    const prefix = outputPrefix.value || 'custom_bundle';
    const freqSuffix = getFreqSuffix(freqSelect.value);
    const newFolder = newFolderInput.value.trim();
    const selectedFolder = folderSelect.value;
    const folder = newFolder || selectedFolder;
    
    const filename = `${prefix}${freqSuffix}_bundle.parquet`;
    const display = folder ? `${folder}/${filename}` : filename;
    filenamePreview.textContent = `Output: ${display}`;
}

async function loadAssets() {
    const freq = freqSelect.value;
    const productType = productTypeSelect.value;
    
    assetsGrid.innerHTML = '<div class="loading">Loading assets...</div>';
    assetsPanel.style.display = 'block';
    
    try {
        const url = `/api/assets/list?freq=${encodeURIComponent(freq)}${productType ? '&product_type=' + encodeURIComponent(productType) : ''}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        renderAssets(data.assets_by_type);
        optionsPanel.style.display = 'block';
    } catch (error) {
        assetsGrid.innerHTML = `<div class="empty-state">Error loading assets: ${error.message}</div>`;
        console.error('Load assets error:', error);
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

freqSelect.addEventListener('change', updateFilenamePreview);
outputPrefix.addEventListener('input', updateFilenamePreview);
folderSelect.addEventListener('change', () => {
    if (folderSelect.value) {
        newFolderInput.value = '';
    }
    updateFilenamePreview();
});
newFolderInput.addEventListener('input', () => {
    if (newFolderInput.value) {
        folderSelect.value = '';
    }
    updateFilenamePreview();
});

createBundleBtn.addEventListener('click', async () => {
    if (selectedAssets.length === 0) return;
    
    createBundleBtn.disabled = true;
    progressPanel.style.display = 'block';
    resultPanel.style.display = 'none';
    
    const assets = selectedAssets.map(a => a.symbol);
    const productTypes = selectedAssets.map(a => a.product_type);
    const selectedConfig = featureConfigSelect.value || null;
    const newFolder = newFolderInput.value.trim();
    const selectedFolder = folderSelect.value;
    const folder = newFolder || selectedFolder || null;
    
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
                folder: folder,
            })
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
            const response = await fetch(`/api/bundle/task/${currentTaskId}/status`);
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
        const response = await fetch(`/api/bundle/task/${currentTaskId}/result`);
        const result = await response.json();
        
        if (result.success) {
            showResult(true, `
                <strong>Bundle created successfully!</strong><br>
                <br>
                <strong>Path:</strong> ${result.relative_path || result.output_path}<br>
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
    loadFolders();
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
            const folderDisplay = bundle.folder ? `<div class="bundle-folder">${bundle.folder}/</div>` : '';
            html += `
                <div class="bundle-item">
                    <div>
                        ${folderDisplay}
                        <div class="bundle-name">${bundle.filename}</div>
                        <div class="bundle-info">${bundle.size_mb} MB | Modified: ${new Date(bundle.modified).toLocaleString()}</div>
                    </div>
                    <div class="bundle-actions">
                        <button class="btn-view" onclick="viewBundle('${bundle.path}')">View</button>
                        <button class="btn-delete" onclick="confirmDelete('${bundle.path}', '${bundle.filename}')">Delete</button>
                    </div>
                </div>
            `;
        });
        bundlesList.innerHTML = html;
    } catch (error) {
        bundlesList.innerHTML = `<div class="empty-state">Error loading bundles: ${error.message}</div>`;
    }
}

window.viewBundle = async function(path) {
    modalTitle.textContent = path.split('/').pop();
    modalBody.innerHTML = '<div class="loading">Loading bundle details...</div>';
    bundleModal.classList.add('active');
    
    try {
        const response = await fetch(`/api/bundle/by-path/${encodeURIComponent(path)}/details`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const details = await response.json();
        
        const startDate = details.start_date ? new Date(details.start_date).toLocaleString() : 'N/A';
        const endDate = details.end_date ? new Date(details.end_date).toLocaleString() : 'N/A';
        
        let assetsHtml = details.assets.map(a => `<span class="tag">${a}</span>`).join('');
        let featuresHtml = details.features.slice(0, 50).map(f => `<span class="tag tag-feature">${f}</span>`).join('');
        if (details.features.length > 50) {
            featuresHtml += `<span class="tag tag-feature">...and ${details.features.length - 50} more</span>`;
        }
        
        modalBody.innerHTML = `
            <div class="detail-grid">
                <div class="detail-stat">
                    <div class="detail-stat-value">${details.total_rows?.toLocaleString()}</div>
                    <div class="detail-stat-label">Total Rows</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${details.total_columns}</div>
                    <div class="detail-stat-label">Columns</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${details.file_size_mb} MB</div>
                    <div class="detail-stat-label">File Size</div>
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-label">Date Range</div>
                <div class="detail-value">
                    <strong>Start:</strong> ${startDate}<br>
                    <strong>End:</strong> ${endDate}
                </div>
            </div>
            
            <div class="detail-section">
                <div class="detail-label">Assets (${details.assets.length})</div>
                <div class="tag-list">${assetsHtml}</div>
            </div>
            
            <div class="detail-section">
                <div class="detail-label">Features (${details.features.length})</div>
                <div class="tag-list">${featuresHtml}</div>
            </div>
        `;
    } catch (error) {
        modalBody.innerHTML = `<div class="empty-state">Error loading bundle details: ${error.message}</div>`;
    }
};

window.confirmDelete = function(path, filename) {
    modalTitle.textContent = 'Confirm Delete';
    modalBody.innerHTML = `
        <p style="margin-bottom: 20px; color: #e0e0e0;">Are you sure you want to delete this bundle?</p>
        <p style="font-family: monospace; color: #6ee7b7; background: rgba(20,20,30,0.6); padding: 12px; border-radius: 6px;">${path}</p>
        <div class="modal-confirm">
            <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
            <button class="btn btn-danger" onclick="deleteBundle('${path}')">Delete</button>
        </div>
    `;
    bundleModal.classList.add('active');
};

window.closeModal = function() {
    bundleModal.classList.remove('active');
};

window.deleteBundle = async function(path) {
    modalBody.innerHTML = '<div class="loading">Deleting...</div>';
    
    try {
        const response = await fetch(`/api/bundle/by-path/${encodeURIComponent(path)}`, {
            method: 'DELETE',
        });
        
        const data = await response.json();
        
        if (response.ok) {
            modalBody.innerHTML = `
                <div class="result-success" style="margin: 0;">
                    <strong>Bundle deleted successfully!</strong><br>
                    ${path}
                </div>
            `;
            setTimeout(() => {
                bundleModal.classList.remove('active');
                loadBundles();
                loadFolders();
            }, 1500);
        } else {
            modalBody.innerHTML = `<div class="result-error" style="margin: 0;">Error: ${data.detail || 'Failed to delete'}</div>`;
        }
    } catch (error) {
        modalBody.innerHTML = `<div class="result-error" style="margin: 0;">Error: ${error.message}</div>`;
    }
};

modalClose.addEventListener('click', () => {
    bundleModal.classList.remove('active');
});

bundleModal.addEventListener('click', (e) => {
    if (e.target === bundleModal) {
        bundleModal.classList.remove('active');
    }
});

async function loadFeatureConfigs() {
    try {
        const response = await fetch('/api/bundle/feature-configs');
        const configs = await response.json();
        
        let html = '<option value="">Default (all features)</option>';
        configs.forEach(config => {
            html += `<option value="${config.filename}">${config.filename}</option>`;
        });
        featureConfigSelect.innerHTML = html;
    } catch (error) {
        featureConfigSelect.innerHTML = '<option value="">Default (all features)</option>';
        console.error('Error loading feature configs:', error);
    }
}

async function loadFolders() {
    try {
        const response = await fetch('/api/bundle/folders');
        const folders = await response.json();
        
        let html = '<option value="">Root</option>';
        folders.forEach(folder => {
            html += `<option value="${folder.path}">${folder.path} (${folder.bundle_count})</option>`;
        });
        folderSelect.innerHTML = html;
    } catch (error) {
        folderSelect.innerHTML = '<option value="">Root</option>';
        console.error('Error loading folders:', error);
    }
}

loadBundles();
loadFeatureConfigs();
loadFolders();
updateFilenamePreview();

// Tab switching functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        tab.classList.add('active');
        const tabId = `tab-${tab.dataset.tab}`;
        document.getElementById(tabId).classList.add('active');
        
        // Resize seasonality charts when switching to that tab
        if (tab.dataset.tab === 'seasonality' && window.seasonalityState) {
            setTimeout(() => {
                Object.values(window.seasonalityState.charts).forEach(chart => {
                    if (chart) chart.resize();
                });
            }, 100);
        }
    });
});
