const state = {
    workers: [],
    jobs: [],
    stats: null,
    currentTab: 'queue',
    refreshInterval: null,
};

const API_BASE = '';

document.addEventListener('DOMContentLoaded', init);

function init() {
    setupTabs();
    setupButtons();
    loadAll();
    startAutoRefresh();
}

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.currentTab = tab.dataset.tab;
            renderJobs();
        });
    });
}

function setupButtons() {
    document.getElementById('btn-refresh').addEventListener('click', loadAll);
    document.getElementById('btn-submit-job').addEventListener('click', openSubmitModal);
    document.getElementById('dispatch-mode').addEventListener('change', toggleManualWorker);
}

function startAutoRefresh() {
    state.refreshInterval = setInterval(loadAll, 5000);
}

function stopAutoRefresh() {
    if (state.refreshInterval) {
        clearInterval(state.refreshInterval);
    }
}

async function loadAll() {
    await Promise.all([loadWorkers(), loadJobs(), loadStats()]);
    updateLastUpdate();
}

async function loadWorkers() {
    try {
        const response = await fetch(`${API_BASE}/api/workers`);
        state.workers = await response.json();
        renderWorkers();
        updateWorkerSelect();
    } catch (e) {
        console.error('Failed to load workers:', e);
    }
}

async function loadJobs() {
    try {
        const response = await fetch(`${API_BASE}/api/jobs?limit=200`);
        state.jobs = await response.json();
        renderJobs();
        updateCounts();
    } catch (e) {
        console.error('Failed to load jobs:', e);
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        state.stats = await response.json();
        renderStats();
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

function renderWorkers() {
    const container = document.getElementById('workers-list');
    
    if (state.workers.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">-</div><p>No workers registered</p></div>';
        return;
    }
    
    container.innerHTML = state.workers.map(worker => `
        <div class="worker-card ${worker.status}">
            <div class="worker-header">
                <span class="worker-name">${escapeHtml(worker.name)}</span>
                <span class="worker-status ${worker.status}">${worker.status}</span>
            </div>
            ${worker.label ? `<div class="worker-label">${escapeHtml(worker.label)}</div>` : ''}
            <div class="worker-info">
                <div>Jobs: ${worker.current_jobs}/${worker.max_concurrent_jobs}</div>
                <div>Completed: ${worker.jobs_completed} | Failed: ${worker.jobs_failed}</div>
                <div>${worker.host}:${worker.port}</div>
            </div>
        </div>
    `).join('');
}

function renderJobs() {
    const container = document.getElementById('jobs-content');
    
    let filteredJobs = state.jobs;
    if (state.currentTab !== 'all') {
        filteredJobs = state.jobs.filter(job => {
            if (state.currentTab === 'queue') return job.status === 'queued';
            if (state.currentTab === 'running') return job.status === 'running';
            if (state.currentTab === 'completed') return job.status === 'completed';
            if (state.currentTab === 'failed') return job.status === 'failed';
            return true;
        });
    }
    
    if (filteredJobs.length === 0) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">-</div><p>No jobs</p></div>';
        return;
    }
    
    container.innerHTML = filteredJobs.map(job => renderJobCard(job)).join('');
}

function renderJobCard(job) {
    const submittedAt = job.submitted_at ? formatDateTime(job.submitted_at) : '-';
    const completedAt = job.completed_at ? formatDateTime(job.completed_at) : '-';
    const duration = job.duration_seconds ? formatDuration(job.duration_seconds) : '-';
    
    let actions = '';
    if (job.status === 'queued' || job.status === 'running') {
        actions += `<button class="btn btn-sm btn-danger" onclick="cancelJob('${job.job_id}')">Cancel</button>`;
    }
    if (job.status === 'completed' || job.status === 'failed') {
        actions += `<button class="btn btn-sm btn-secondary" onclick="viewOutput('${job.job_id}')">View Output</button>`;
        actions += `<button class="btn btn-sm btn-primary" onclick="rerunJob('${job.job_id}')">Re-run</button>`;
    }
    if (job.status !== 'running') {
        actions += `<button class="btn btn-sm btn-secondary" onclick="deleteJob('${job.job_id}')">Delete</button>`;
    }
    
    return `
        <div class="job-card ${job.status}">
            <div class="job-header">
                <div>
                    <div class="job-title">${escapeHtml(job.name)}</div>
                    <div class="job-id">${job.job_id}</div>
                </div>
                <span class="priority-badge ${job.priority}">${job.priority.toUpperCase()}</span>
            </div>
            <div class="job-meta">
                <div class="job-meta-item">Status: <strong>${job.status}</strong></div>
                <div class="job-meta-item">Worker: ${job.worker_name || '-'}</div>
                <div class="job-meta-item">Submitted: ${submittedAt}</div>
                ${job.status === 'completed' || job.status === 'failed' ? `
                    <div class="job-meta-item">Completed: ${completedAt}</div>
                    <div class="job-meta-item">Duration: ${duration}</div>
                ` : ''}
            </div>
            <div class="job-actions">${actions}</div>
        </div>
    `;
}

function renderStats() {
    if (!state.stats) return;
    
    const container = document.getElementById('header-stats');
    container.innerHTML = `
        <div class="stat"><span>Workers:</span> <span class="stat-value">${state.stats.online_workers}/${state.stats.total_workers}</span></div>
        <div class="stat"><span>Running:</span> <span class="stat-value">${state.stats.running_jobs}</span></div>
        <div class="stat"><span>Queued:</span> <span class="stat-value">${state.stats.queued_jobs}</span></div>
        <div class="stat"><span>Today:</span> <span class="stat-value">${state.stats.jobs_today}</span></div>
    `;
}

function updateCounts() {
    const counts = {
        queue: state.jobs.filter(j => j.status === 'queued').length,
        running: state.jobs.filter(j => j.status === 'running').length,
    };
    
    document.getElementById('queue-count').textContent = counts.queue;
    document.getElementById('running-count').textContent = counts.running;
}

function updateLastUpdate() {
    document.getElementById('last-update').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
}

function updateWorkerSelect() {
    const select = document.getElementById('target-worker');
    select.innerHTML = '<option value="">Select worker...</option>' +
        state.workers.filter(w => w.status !== 'offline').map(w => 
            `<option value="${w.worker_id}">${w.name} (${w.current_jobs}/${w.max_concurrent_jobs})</option>`
        ).join('');
}

function toggleManualWorker() {
    const mode = document.getElementById('dispatch-mode').value;
    const manualSelect = document.getElementById('manual-worker-select');
    manualSelect.classList.toggle('hidden', mode !== 'manual');
}

function openSubmitModal() {
    document.getElementById('submit-modal').classList.remove('hidden');
    document.getElementById('job-name').value = '';
    document.getElementById('job-priority').value = 'normal';
    document.getElementById('dispatch-mode').value = 'auto';
    document.getElementById('target-worker').value = '';
    document.getElementById('job-script').value = '';
    document.getElementById('job-config').value = '{}';
    document.getElementById('manual-worker-select').classList.add('hidden');
}

function closeSubmitModal() {
    document.getElementById('submit-modal').classList.add('hidden');
}

async function submitJob() {
    const name = document.getElementById('job-name').value.trim();
    const priority = document.getElementById('job-priority').value;
    const dispatchMode = document.getElementById('dispatch-mode').value;
    const targetWorker = document.getElementById('target-worker').value;
    const script = document.getElementById('job-script').value;
    const configStr = document.getElementById('job-config').value;
    
    if (!name) {
        alert('Please enter a job name');
        return;
    }
    
    if (!script) {
        alert('Please enter a script');
        return;
    }
    
    let config = {};
    try {
        config = JSON.parse(configStr);
    } catch (e) {
        alert('Invalid JSON in config');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                priority,
                dispatch_mode: dispatchMode,
                target_worker_id: dispatchMode === 'manual' ? targetWorker : null,
                script,
                config,
            }),
        });
        
        if (response.ok) {
            closeSubmitModal();
            loadAll();
            setStatus('Job submitted successfully');
        } else {
            const error = await response.json();
            alert('Failed to submit job: ' + (error.detail || 'Unknown error'));
        }
    } catch (e) {
        alert('Error submitting job: ' + e.message);
    }
}

async function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}/cancel`, { method: 'POST' });
        if (response.ok) {
            loadAll();
            setStatus('Job cancelled');
        }
    } catch (e) {
        alert('Error cancelling job: ' + e.message);
    }
}

async function deleteJob(jobId) {
    if (!confirm('Are you sure you want to delete this job?')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: 'DELETE' });
        if (response.ok) {
            loadAll();
            setStatus('Job deleted');
        }
    } catch (e) {
        alert('Error deleting job: ' + e.message);
    }
}

async function rerunJob(jobId) {
    try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}/rerun`, { method: 'POST' });
        if (response.ok) {
            loadAll();
            setStatus('Job re-submitted');
        }
    } catch (e) {
        alert('Error re-running job: ' + e.message);
    }
}

async function viewOutput(jobId) {
    try {
        const response = await fetch(`${API_BASE}/api/jobs/${jobId}/output`);
        const data = await response.json();
        
        const job = state.jobs.find(j => j.job_id === jobId);
        
        document.getElementById('modal-title').textContent = `Output: ${job ? job.name : jobId}`;
        document.getElementById('modal-body').innerHTML = `
            <div class="output-viewer">
                <pre>${escapeHtml(data.output || '(no output)')}</pre>
            </div>
        `;
        document.getElementById('modal-overlay').classList.remove('hidden');
    } catch (e) {
        alert('Error loading output: ' + e.message);
    }
}

function closeModal() {
    document.getElementById('modal-overlay').classList.add('hidden');
}

function setStatus(message) {
    document.getElementById('status-message').textContent = message;
    setTimeout(() => {
        document.getElementById('status-message').textContent = 'Ready';
    }, 3000);
}

function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

function formatDuration(seconds) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;')
              .replace(/</g, '&lt;')
              .replace(/>/g, '&gt;')
              .replace(/"/g, '&quot;')
              .replace(/'/g, '&#039;');
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeModal();
        closeSubmitModal();
    }
});

document.getElementById('modal-overlay').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
});

document.getElementById('submit-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeSubmitModal();
});
