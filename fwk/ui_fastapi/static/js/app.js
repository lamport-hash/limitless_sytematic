const state = {
    currentPid: null,
    ws: null,
    isRunning: false,
    modelParams: {},
    workers: [],
    selectedFeatures: [],
    selectedTargets: [],
    availableFeatures: [],
    availableTargets: [],
};

const MODEL_PARAMS_FIELDS = {
    xgb: [
        { key: 'n_estimators', type: 'number', default: 100 },
        { key: 'max_depth', type: 'number', default: 6 },
        { key: 'learning_rate', type: 'number', default: 0.1, step: 0.01 },
        { key: 'subsample', type: 'number', default: 1.0, step: 0.1, min: 0.1, max: 1.0 },
        { key: 'colsample_bytree', type: 'number', default: 1.0, step: 0.1, min: 0.1, max: 1.0 },
        { key: 'min_child_weight', type: 'number', default: 1 }
    ],
    rf_sk: [
        { key: 'n_estimators', type: 'number', default: 100 },
        { key: 'max_depth', type: 'number', default: null, nullable: true },
        { key: 'min_samples_split', type: 'number', default: 2 },
        { key: 'min_samples_leaf', type: 'number', default: 1 },
        { key: 'max_features', type: 'select', default: 'sqrt', options: ['sqrt', 'log2', null] }
    ],
    dt_sk: [
        { key: 'max_depth', type: 'number', default: null, nullable: true },
        { key: 'min_samples_split', type: 'number', default: 2 },
        { key: 'min_samples_leaf', type: 'number', default: 1 },
        { key: 'max_features', type: 'select', default: null, options: ['sqrt', 'log2', null] },
        { key: 'criterion', type: 'select', default: 'squared_error', options: ['squared_error', 'absolute_error', 'friedman_mse', 'gini', 'entropy'] }
    ],
    mlp_torch: [
        { key: 'hidden_sizes', type: 'text', default: '[128, 64]' },
        { key: 'batch_size', type: 'number', default: 32 },
        { key: 'epochs', type: 'number', default: 100 },
        { key: 'learning_rate', type: 'number', default: 0.001, step: 0.0001 },
        { key: 'optimizer', type: 'select', default: 'adam', options: ['adam', 'sgd', 'rmsprop'] },
        { key: 'dropout', type: 'number', default: 0.2, step: 0.1, min: 0, max: 1 },
        { key: 'weight_decay', type: 'number', default: 0.0001, step: 0.00001 },
        { key: 'patience', type: 'number', default: 20 }
    ],
    bnn_gauto: [
        { key: 'n_classes', type: 'number', default: 2 },
        { key: 'kernels', type: 'text', default: '["periodic", "linear", "matern"]' },
        { key: 'epochs', type: 'number', default: 50 },
        { key: 'learning_rate', type: 'number', default: 0.01, step: 0.001 },
        { key: 'patience', type: 'number', default: 10 }
    ],
};

const REGRESSION_METRICS = ['mse', 'rmse', 'mae', 'r2'];
const CLASSIFICATION_METRICS = ['accuracy', 'f1', 'precision', 'recall'];

document.addEventListener('DOMContentLoaded', init);

function init() {
    setupSectionToggles();
    setupRunModeToggle();
    setupDataSourceToggle();
    setupTrainingToggle();
    setupHyperparamToggle();
    setupFeatureSelectionToggle();
    setupFeatureBundlingToggle();
    setupOutputToggle();
    setupFeaturesToggle();
    setupTargetsToggle();
    setupJudgeToggle();
    setupModelChange();
    setupTaskTypeChange();
    setupButtons();
    loadModelParams('xgb');
    loadMetrics('regression');
    loadDataFiles();
    loadWorkers();
    loadFeatureTypes();
    loadTargetFunctions();
    loadAvailableData();
}

function setupSectionToggles() {
    document.querySelectorAll('.section-header').forEach(header => {
        header.addEventListener('click', () => {
            const sectionId = header.dataset.section;
            const content = document.getElementById(`section-${sectionId}`);
            const icon = header.querySelector('.toggle-icon');
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                icon.textContent = '-';
            } else {
                content.classList.add('collapsed');
                icon.textContent = '+';
            }
        });
    });
}
function setupRunModeToggle() {
    const runMode = document.getElementById('run-mode');
    const dispatchJobName = document.getElementById('dispatch-job-name')
    const dispatchPriority = document.getElementById('dispatch-priority')
    const workerSelect = document.getElementById('worker-select')
    const managerLink = document.getElementById('manager-link')
    const btnRun = document.getElementById('btn-run')
    
    runMode.addEventListener('change', () => {
        const mode = runMode.value;
        
        if (mode === 'local') {
            dispatchJobName.style.display = 'none'
            dispatchPriority.style.display = 'none'
            workerSelect.style.display = 'none'
            managerLink.style.display = 'none'
            btnRun.textContent = 'Run'
        } else if (mode === 'manager') {
            dispatchJobName.style.display = 'block'
            dispatchPriority.style.display = 'block'
            workerSelect.style.display = 'none'
            managerLink.style.display = 'block'
            btnRun.textContent = 'Submit'
        } else if (mode === 'manual') {
            dispatchJobName.style.display = 'block'
            dispatchPriority.style.display = 'block'
            workerSelect.style.display = 'block'
            managerLink.style.display = 'block'
            btnRun.textContent = 'Submit'
            loadWorkers()
        }
    })
}
function async function loadWorkers() {
    try {
        const response = await fetch('/api/manager/workers')
        const data = await response.json()
        
        if (data.success && data.workers) {
            state.workers = data.workers
            const select = document.getElementById('target-worker')
            select.innerHTML = '<option value="">Select worker...</option>' +
                data.workers
                    .filter(w => w.status !== 'offline')
                    .map(w => `<option value="${w.worker_id}">${w.name} (${w.current_jobs}/${w.max_concurrent_jobs})</option>`)
                    .join('')
        }
    } catch (e) {
        console.error('Failed to load workers:', e)
    }
}
function setupDataSourceToggle() {
    const sourceType = document.getElementById('source-type')
    const syntheticOptions = document.getElementById('synthetic-options')
    const fileOptions = document.getElementById('file-options')
    const normalizedOptions = document.getElementById('normalized-options')
    const searchBtn = document.getElementById('btn-search-data')
    
    if (searchBtn) {
        searchBtn.addEventListener('click', searchNormalizedData)
    }
    
    sourceType.addEventListener('change', () => {
        const value = sourceType.value
        syntheticOptions.style.display = value === 'synthetic' ? 'block' : 'none'
        fileOptions.style.display = value === 'file' ? 'block' : 'none'
        normalizedOptions.style.display = value === 'normalized' ? 'block' : 'none'
    })
}
function setupTrainingToggle() {
    const splitMode = document.getElementById('split-mode')
    const retrainOptions = document.getElementById('retrain-options')
    const retrainMode = document.getElementById('retrain-mode')
    const slidingOptions = document.getElementById('sliding-options')
    
    splitMode.addEventListener('change', () => {
        if (splitMode.value === 'periodic_retrain') {
            retrainOptions.style.display = 'block'
        } else {
            retrainOptions.style.display = 'none'
        }
    })
    
    retrainMode.addEventListener('change', () => {
        if (retrainMode.value === 'sliding') {
            slidingOptions.style.display = 'block'
        } else {
            slidingOptions.style.display = 'none'
        }
    })
}
function setupHyperparamToggle() {
    const enabled = document.getElementById('hp-enabled')
    const options = document.getElementById('hp-options')
    
    enabled.addEventListener('change', () => {
        options.style.display = enabled.checked ? 'block' : 'none'
    })
}
function setupFeatureSelectionToggle() {
    const enabled = document.getElementById('fs-enabled')
    const options = document.getElementById('fs-options')
    const method = document.getElementById('fs-method')
    const cleverOptions = document.getElementById('fs-clever-options')
    
    enabled.addEventListener('change', () => {
        options.style.display = enabled.checked ? 'block' : 'none'
    })
    
    method.addEventListener('change', () => {
        cleverOptions.style.display = method.value === 'greedy_clever' ? 'block' : 'none'
    })
}
function setupFeatureBundlingToggle() {
    const lagEnabled = document.getElementById('bundle-lag')
    const lagOptions = document.getElementById('lag-options')
    const rollingEnabled = document.getElementById('bundle-rolling')
    const rollingOptions = document.getElementById('rolling-options')
    const interactionEnabled = document.getElementById('bundle-interaction')
    const interactionOptions = document.getElementById('interaction-options')
    
    lagEnabled.addEventListener('change', () => {
        lagOptions.style.display = lagEnabled.checked ? 'block' : 'none'
    })
    
    rollingEnabled.addEventListener('change', () => {
        rollingOptions.style.display = rollingEnabled.checked ? 'block' : 'none'
    })
    
    interactionEnabled.addEventListener('change', () => {
        interactionOptions.style.display = interactionEnabled.checked ? 'block' : 'none'
    })
}
function setupFeaturesToggle() {
    const enabled = document.getElementById('features-enabled')
    const config = document.getElementById('features-config')
    
    enabled.addEventListener('change', () => {
        config.style.display = enabled.checked ? 'block' : 'none'
    })
}
function setupTargetsToggle() {
    const enabled = document.getElementById('targets-enabled')
    const config = document.getElementById('targets-config')
    
    enabled.addEventListener('change', () => {
        config.style.display = enabled.checked ? 'block' : 'none'
    })
}
function setupJudgeToggle() {
    const enabled = document.getElementById('judge-enabled')
    const options = document.getElementById('judge-options')
    
    if (enabled && options) {
        enabled.addEventListener('change', () => {
            options.style.display = enabled.checked ? 'block' : 'none'
        })
    }
}
function setupOutputToggle() {
    const saveModel = document.getElementById('save-model')
    const saveOptions = document.getElementById('save-options')
    
    saveModel.addEventListener('change', () => {
        saveOptions.style.display = saveModel.checked ? 'block' : 'none'
    })
}
function setupModelChange() {
    const modelType = document.getElementById('model-type')
    modelType.addEventListener('change', () => {
        loadModelParams(modelType.value)
    })
}
function setupTaskTypeChange() {
    const taskType = document.getElementById('task-type')
    const averagingGroup = document.getElementById('averaging-group')
    
    taskType.addEventListener('change', () => {
        loadMetrics(taskType.value)
        if (taskType.value === 'classification') {
            averagingGroup.style.display = 'block'
        } else {
            averagingGroup.style.display = 'none'
        }
    })
}
function loadModelParams(modelType) {
    const container = document.getElementById('model-params-container')
    const fields = MODEL_PARAMS_FIELDS[modelType] || []
    
    container.innerHTML = ''
    state.modelParams = {}
    
    fields.forEach(field => {
        const savedValue = state.modelParams[field.key]
        const value = savedValue !== undefined ? savedValue : field.default
        
        const div = document.createElement('div')
        div.className = 'form-group'
        
        const label = document.createElement('label')
        label.textContent = field.key
        div.appendChild(label)

        if (field.type === 'select') {
            const select = document.createElement('select')
            select.id = `param-${field.key}`
            field.options.forEach(opt => {
                const option = document.createElement('option')
                option.value = opt === null ? 'null' : opt
                option.textContent = opt === null ? 'None' : opt
                if (opt === value) option.selected = true
                select.appendChild(option)
            })
            select.addEventListener('change', () => {
                const v = select.value === 'null' ? null : select.value
                state.modelParams[field.key] = v
            })
            state.modelParams[field.key] = value
            div.appendChild(select)
        } else {
            const input = document.createElement('input')
            input.type = field.type
            input.id = `param-${field.key}`
            input.value = value ?? ''
            if (field.step) input.step = field.step
            if (field.min !== undefined) input.min = field.min
            if (field.max !== undefined) input.max = field.max
            
            input.addEventListener('change', () => {
                if (field.type === 'number') {
                    state.modelParams[field.key] = input.value ? parseFloat(input.value) : null
                } else if (field.key === 'hidden_sizes' || field.key === 'kernels') {
                    try {
                        state.modelParams[field.key] = JSON.parse(input.value)
                    } catch (e) {
                        state.modelParams[field.key] = input.value
                    }
                } else {
                    state.modelParams[field.key] = input.value
                }
            })
            state.modelParams[field.key] = value
            div.appendChild(input)
        }

        container.appendChild(div)
    })
}
function loadMetrics(taskType) {
    const select = document.getElementById('primary-metric')
    const metrics = taskType === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS
    
    select.innerHTML = ''
    metrics.forEach(m => {
        const option = document.createElement('option')
        option.value = m
        option.textContent = m.toUpperCase()
        select.appendChild(option)
    })
}
async function loadDataFiles() {
    try {
        const response = await fetch('/api/data-files')
        const data = await response.json()
        const select = document.getElementById('data-file')
        
        select.innerHTML = '<option value="">Select a file...</option>'
        data.files.forEach(file => {
            const option = document.createElement('option')
            option.value = file.path
            option.textContent = file.name
            select.appendChild(option)
        })
    } catch (e) {
        console.error('Failed to load data files:', e)
    }
}
async function loadFeatureTypes() {
    try {
        const response = await fetch('/api/features/types')
        const data = await response.json()
        state.availableFeatures = data.types || []
        
        const select = document.getElementById('feature-type-select')
        select.innerHTML = '<option value="">Select feature...</option>'
        state.availableFeatures.forEach(f => {
            const option = document.createElement('option')
            option.value = f.name
            option.textContent = f.name
            select.appendChild(option)
        })
    } catch (e) {
        console.error('Failed to load feature types:', e)
    }
}
async function loadTargetFunctions() {
    try {
        const response = await fetch('/api/targets/functions')
        const data = await response.json()
        state.availableTargets = data.functions || []
        
        const select = document.getElementById('target-function-select')
        select.innerHTML = '<option value="">Select function...</option>'
        state.availableTargets.forEach(f => {
            const option = document.createElement('option')
            option.value = f.name
            option.textContent = f.name
            select.appendChild(option)
        })
    } catch (e) {
        console.error('Failed to load target functions:', e)
    }
}
async function loadAvailableData() {
    try {
        const response = await fetch('/api/data/available')
        const data = await response.json()
        
        if (data.frequencies) {
            const freqSelect = document.getElementById('norm-data-freq')
            freqSelect.innerHTML = '<option value="">All</option>'
            data.frequencies.forEach(f => {
                const option = document.createElement('option')
                option.value = f
                option.textContent = f
                freqSelect.appendChild(option)
            })
        }
        
        if (data.sources) {
            const sourceSelect = document.getElementById('norm-source')
            sourceSelect.innerHTML = '<option value="">All</option>'
            data.sources.forEach(s => {
                const option = document.createElement('option')
                option.value = s
                option.textContent = s
                sourceSelect.appendChild(option)
            })
        }
    } catch (e) {
        console.error('Failed to load available data:', e)
    }
}
async function searchNormalizedData() {
    const params = new URLSearchParams()
    
    const symbol = document.getElementById('norm-symbol').value
    const dataFreq = document.getElementById('norm-data-freq').value
    const source = document.getElementById('norm-source').value
    const productType = document.getElementById('norm-product-type').value
    const start = document.getElementById('norm-start').value
    const end = document.getElementById('norm-end').value
    
    if (symbol) params.append('symbol', symbol)
    if (dataFreq) params.append('data_freq', dataFreq)
    if (source) params.append('source', source)
    if (productType) params.append('product_type', productType)
    if (start) params.append('start', start)
    if (end) params.append('end', end)
    
    document.getElementById('search-status').textContent = 'Searching...'
    
    try {
        const response = await fetch(`/api/data/search?${params.toString()}`)
        const data = await response.json()
        
        const select = document.getElementById('norm-file-select')
        select.innerHTML = '<option value="">-- Select a file --</option>'
        
        if (data.files && data.files.length > 0) {
            data.files.forEach(f => {
                const option = document.createElement('option')
                option.value = f.path
                option.textContent = `${f.instrument} | ${f.data_freq} | ${f.source}`
                option.dataset.fileInfo = JSON.stringify(f)
                select.appendChild(option)
            })
            document.getElementById('search-status').textContent = `Found ${data.files.length} files`
        } else {
            document.getElementById('search-status').textContent = 'No files found'
        }
    } catch (e) {
        document.getElementById('search-status').textContent = 'Error: ' + e.message
    }
}
function addFeature() {
    const typeSelect = document.getElementById('feature-type-select')
    const periodsInput = document.getElementById('feature-periods')
    
    const featureType = typeSelect.value
    if (!featureType) return
    
    const periods = periodsInput.value
        ? periodsInput.value.split(',').map(p => parseInt(p.trim())).filter(p => !isNaN(p))
        : null
    
    state.selectedFeatures.push({
        feature_type: featureType,
        periods: periods
    })
    
    updateFeaturesList()
}
function addTarget() {
    const funcSelect = document.getElementById('target-function-select')
    const nPeriodsInput = document.getElementById('target-n-periods')
    
    const func = funcSelect.value
    if (!func) return
    
    const nPeriods = parseInt(nPeriodsInput.value) || 60
    
    state.selectedTargets.push({
        function: func,
        params: { N_periods: nPeriods }
    })
    
    updateTargetsList()
}
function updateFeaturesList() {
    const list = document.getElementById('features-list')
    list.innerHTML = ''
    
    state.selectedFeatures.forEach((f, i) => {
        const div = document.createElement('div')
        div.className = 'item-tag'
        div.innerHTML = `
            <span>${f.feature_type}${f.periods ? ` (${f.periods.join(', ')})` : ''}</span>
            <button class="btn-remove" data-index="${i}">×</button>
        `
        list.appendChild(div)
    })
    
    list.querySelectorAll('.btn-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            state.selectedFeatures.splice(parseInt(btn.dataset.index), 1)
            updateFeaturesList()
        })
    })
}
function updateTargetsList() {
    const list = document.getElementById('targets-list')
    list.innerHTML = ''
    
    state.selectedTargets.forEach((t, i) => {
        const div = document.createElement('div')
        div.className = 'item-tag'
        div.innerHTML = `
            <span>${t.function} (N=${t.params.N_periods || 60})</span>
            <button class="btn-remove" data-index="${i}">×</button>
        `
        list.appendChild(div)
    })
    
    list.querySelectorAll('.btn-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            state.selectedTargets.splice(parseInt(btn.dataset.index), 1)
            updateTargetsList()
        })
    })
}
function setupButtons() {
    document.getElementById('btn-generate').addEventListener('click', generateScript)
    document.getElementById('btn-run').addEventListener('click', runScript)
    document.getElementById('btn-stop').addEventListener('click', stopScript)
    document.getElementById('btn-save').addEventListener('click', saveScript)
    document.getElementById('btn-clear-output').addEventListener('click', clearOutput)
    
    const addFeatureBtn = document.getElementById('btn-add-feature')
    if (addFeatureBtn) {
        addFeatureBtn.addEventListener('click', addFeature)
    }
    
    const addTargetBtn = document.getElementById('btn-add-target')
    if (addTargetBtn) {
        addTargetBtn.addEventListener('click', addTarget)
    }
}
function collectConfig() {
    const taskType = document.getElementById('task-type').value
    const modelType = document.getElementById('model-type').value
    const sourceType = document.getElementById('source-type').value
    
    const config = {
        task_type: taskType,
        model_type: modelType,
        model_params: { ...state.modelParams },
        training: {
            mode: document.getElementById('split-mode').value,
            train_ratio: parseFloat(document.getElementById('train-ratio').value),
            val_ratio: parseFloat(document.getElementById('val-ratio').value),
            normalization: document.getElementById('normalization').value,
            normalize_targets: document.getElementById('normalize-targets').checked
        },
        metrics: {
            primary_metric: document.getElementById('primary-metric').value,
            averaging: document.getElementById('averaging').value
        },
        hyperparameter_tuning: {
            enabled: document.getElementById('hp-enabled').checked,
            strategy: document.getElementById('hp-strategy').value,
            n_random_samples: parseInt(document.getElementById('hp-samples').value)
        },
        feature_selection: {
            enabled: document.getElementById('fs-enabled').checked,
            method: document.getElementById('fs-method').value,
            max_features: parseInt(document.getElementById('fs-max').value),
            min_features: parseInt(document.getElementById('fs-min').value),
            greedy_clever_p: parseInt(document.getElementById('fs-clever-p').value),
            greedy_clever_n: parseInt(document.getElementById('fs-clever-n').value),
            greedy_clever_m: 10
        },
        feature_bundling: {
            lag: {
                enabled: document.getElementById('bundle-lag').checked,
                lags: document.getElementById('lag-values').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
            },
            rolling: {
                enabled: document.getElementById('bundle-rolling').checked,
                windows: document.getElementById('rolling-windows').value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)),
                ops: Array.from(document.getElementById('rolling-ops').selectedOptions).map(o => o.value)
            },
            interaction: {
                enabled: document.getElementById('bundle-interaction').checked,
                max_interactions: parseInt(document.getElementById('interaction-max').value)
            }
        },
        data_source: collectDataSourceConfig(sourceType),
        features: {
            enabled: document.getElementById('features-enabled').checked,
            features: state.selectedFeatures,
            config_file: document.getElementById('features-config-file').value || null
        },
        targets: {
            enabled: document.getElementById('targets-enabled').checked,
            asset: document.getElementById('target-asset').value || null,
            targets: state.selectedTargets
            config_file: document.getElementById('targets-config-file').value || null
        },
        judge: {
            enabled: document.getElementById('judge-enabled')?.checked ?? true,
            generate_prompt: document.getElementById('judge-generate-prompt')?.checked ?? false
        },
        verbose: document.getElementById('verbose').checked,
        save_model: document.getElementById('save-model').checked,
        model_save_path: document.getElementById('save-path').value,
        model_id: document.getElementById('model-id').value
    }
    
    if (document.getElementById('split-mode').value === 'periodic_retrain') {
        config.training.retrain_period = parseInt(document.getElementById('retrain-period').value)
        config.training.retrain_mode = document.getElementById('retrain-mode').value
        if (document.getElementById('retrain-mode').value === 'sliding') {
            config.training.sliding_window_size = parseInt(document.getElementById('sliding-window').value)
        }
    }
    
    return config
}
function collectDataSourceConfig(sourceType) {
    const dsConfig = {
        source_type: sourceType,
        target_column: 'target',
        feature_columns: null,
    }
    
    if (sourceType === 'synthetic') {
        dsConfig.use_synthetic = true
        dsConfig.synthetic_n_samples = parseInt(document.getElementById('synthetic-samples').value)
        dsConfig.synthetic_n_features = parseInt(document.getElementById('synthetic-features').value)
    } else if (sourceType === 'file') {
        dsConfig.file_path = document.getElementById('data-file').value || null
        dsConfig.target_column = document.getElementById('target-column').value
    } else if (sourceType === 'normalized') {
        const fileSelect = document.getElementById('norm-file-select')
        const selectedOption = fileSelect.options[fileSelect.selectedIndex]
        
        if (selectedOption && selectedOption.dataset.fileInfo) {
            const fileInfo = JSON.parse(selectedOption.dataset.fileInfo)
            dsConfig.file_path = fileInfo.path
            dsConfig.symbol = fileInfo.instrument
            dsConfig.data_freq = fileInfo.data_freq
            dsConfig.source = fileInfo.source
            dsConfig.product_type = fileInfo.product_type
        }
        dsConfig.target_column = document.getElementById('norm-target-column').value
        
    }
    
    return dsConfig
}
async function generateScript() {
    const config = collectConfig()
    
    try {
        const response = await fetch('/api/generate-script', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        })
        const data = await response.json()
        
        if (data.success) {
            const codeEl = document.getElementById('script-content')
            codeEl.textContent = data.script
            Prism.highlightElement(codeEl)
            document.getElementById('script-status').textContent = 'Script generated'
        } else {
            document.getElementById('script-status').textContent = 'Error: ' + data.error
        }
    } catch (e) {
        document.getElementById('script-status').textContent = 'Error: ' + e.message
    }
}
async function runScript() {
    if (state.isRunning) return
    
    const runMode = document.getElementById('run-mode').value
    
    if (runMode === 'local') {
        await runLocalScript()
    } else {
        await submitToManager()
    }
}
async function runLocalScript() {
    const config = collectConfig()
    const btnRun = document.getElementById('btn-run')
    const btnStop = document.getElementById('btn-stop')
    const outputEl = document.getElementById('output-content')
    
    outputEl.textContent = 'Starting...\n'
    outputEl.parentElement.classList.add('running')
    outputEl.parentElement.classList.remove('error')
    
    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        })
        const data = await response.json()
        
        if (data.success && data.pid) {
            state.currentPid = data.pid
            state.isRunning = true
            btnRun.disabled = true
            btnStop.disabled = false
            document.getElementById('script-status').textContent = `Running (PID: ${data.pid})`
            
            connectWebSocket(data.pid)
        } else {
            outputEl.textContent += 'Error: ' + (data.error || 'Failed to start process')
            outputEl.parentElement.classList.add('error')
        }
    } catch (e) {
        outputEl.textContent += 'Error: ' + e.message
        outputEl.parentElement.classList.add('error')
    }
}
async function submitToManager() {
    const config = collectConfig()
    const outputEl = document.getElementById('output-content')
    const runMode = document.getElementById('run-mode').value
    
    const jobName = document.getElementById('job-name').value || `${config.model_type}_${config.task_type}`
    const priority = document.getElementById('job-priority').value
    const dispatchMode = runMode === 'manual' ? 'manual' : 'auto'
    const targetWorker = document.getElementById('target-worker').value
    
    outputEl.textContent = 'Submitting job to manager...\n'
    outputEl.parentElement.classList.remove('running', 'error')
    
    try {
        const params = new URLSearchParams({
            name: jobName,
            priority: priority,
            dispatch_mode: dispatchMode,
        })
        
        if (dispatchMode === 'manual' && targetWorker) {
            params.append('target_worker_id', targetWorker)
        }
        
        const response = await fetch(`/api/manager/submit?${params.toString()}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        })
        const data = await response.json()
        
        if (data.success && data.job) {
            outputEl.textContent += `Job submitted successfully!\n`
            outputEl.textContent += `Job ID: ${data.job.job_id}\n`
            outputEl.textContent += `Status: ${data.job.status}\n`
            outputEl.textContent += `\nView in Manager UI: http://localhost:8888\n`
            document.getElementById('script-status').textContent = `Job ${data.job.job_id} submitted`
        } else {
            outputEl.textContent += 'Error: ' + (data.error || 'Failed to submit job')
            outputEl.parentElement.classList.add('error')
        }
    } catch (e) {
        outputEl.textContent += 'Error: ' + e.message
        outputEl.parentElement.classList.add('error')
    }
}
function connectWebSocket(pid) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    state.ws = new WebSocket(`${protocol}//${window.location.host}/ws/stream/${pid}`)
    
    const outputEl = document.getElementById('output-content')
    
    state.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        
        if (msg.type === 'output') {
            outputEl.textContent += msg.data
            outputEl.scrollTop = outputEl.scrollHeight
        } else if (msg.type === 'complete') {
            outputEl.parentElement.classList.remove('running')
            if (msg.return_code === 0) {
                outputEl.textContent += '\n--- Completed successfully ---\n'
            } else {
                outputEl.textContent += `\n--- Completed with return code ${msg.return_code} ---\n`
                outputEl.parentElement.classList.add('error')
            }
            finishRun()
        } else if (msg.type === 'error') {
            outputEl.textContent += '\nError: ' + msg.data + '\n'
            outputEl.parentElement.classList.add('error')
            finishRun()
        }
    })
    
    state.ws.onerror = () => {
        outputEl.textContent += '\nWebSocket error\n'
        outputEl.parentElement.classList.add('error')
        finishRun()
    })
    
    state.ws.onclose = () => {
        if (state.isRunning) {
            finishRun()
        }
    }
}
function finishRun() {
    state.isRunning = false
    document.getElementById('btn-run').disabled = false
    document.getElementById('btn-stop').disabled = true
    document.getElementById('script-status').textContent = 'Ready'
    
    if (state.ws) {
        state.ws.close()
        state.ws = null
    }
}
async function stopScript() {
    if (!state.currentPid) return
    
    try {
        await fetch(`/api/stop/${state.currentPid}`, { method: 'POST' })
        document.getElementById('output-content').textContent += '\n--- Process stopped ---\n'
        finishRun()
    } catch (e) {
        console.error('Failed to stop process:', e)
    }
}
function saveScript() {
    const script = document.getElementById('script-content').textContent
    const blob = new Blob([script], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'fitting_script.py'
    a.click()
    URL.revokeObjectURL(url)
}
function clearOutput() {
    document.getElementById('output-content').textContent = 'Ready to run...\n'
    document.getElementById('output-content').parentElement.classList.remove('running', 'error')
}
