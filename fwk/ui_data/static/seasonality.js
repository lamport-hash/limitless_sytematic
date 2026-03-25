/**
 * Seasonality Analysis Module
 * Handles all seasonality analysis functionality including charts and interactions.
 */

const SeasonalityState = {
    asset: null,
    metric: 'returns',
    interval: 60,
    customWindow: null,
    customWindowEnabled: false,
    dateRange: '30y',
    startDate: null,
    endDate: null,
    drillDown: null,
    analysisData: null,
    charts: {},
};

const WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 
                     'July', 'August', 'September', 'October', 'November', 'December'];

const METRIC_COLORS = {
    returns: '#22D3EE',
    volatility: '#F97316',
    rsi: '#A855F7',
};

const POSITIVE_COLOR = '#10B981';
const NEGATIVE_COLOR = '#EF4444';

document.addEventListener('DOMContentLoaded', () => {
    initSeasonalityModule();
});

function initSeasonalityModule() {
    initAssetSearch();
    initMetricToggle();
    initTimeframeSelection();
    initDateRange();
    initChartTabs();
    initDataTable();
    initRunAnalysis();
}

function initAssetSearch() {
    const searchInput = document.getElementById('seasonalityAssetSearch');
    const dropdown = document.getElementById('seasonalityAssetDropdown');
    let debounceTimer;
    
    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const query = e.target.value.trim();
        
        if (query.length < 1) {
            dropdown.classList.remove('active');
            return;
        }
        
        debounceTimer = setTimeout(() => searchAssets(query), 300);
    });
    
    searchInput.addEventListener('focus', (e) => {
        if (e.target.value.trim().length > 0) {
            searchAssets(e.target.value.trim());
        }
    });
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.asset-selector-container')) {
            dropdown.classList.remove('active');
        }
    });
}

async function searchAssets(query) {
    const dropdown = document.getElementById('seasonalityAssetDropdown');
    
    try {
        const response = await fetch(`/api/seasonality/assets/search?q=${encodeURIComponent(query)}&limit=10`);
        const assets = await response.json();
        
        if (assets.length === 0) {
            dropdown.innerHTML = '<div class="asset-dropdown-item"><span class="asset-symbol">No assets found</span></div>';
        } else {
            dropdown.innerHTML = assets.map(asset => `
                <div class="asset-dropdown-item" data-symbol="${asset.symbol}" data-frequencies='${JSON.stringify(asset.frequencies)}' data-default-freq="${asset.default_freq}">
                    <span class="asset-symbol">${asset.symbol}</span>
                    <span class="asset-freqs">${asset.frequencies.map(f => f.replace('candle_', '')).join(', ')}</span>
                    <span class="asset-type">${asset.product_type}</span>
                </div>
            `).join('');
            
            dropdown.querySelectorAll('.asset-dropdown-item').forEach(item => {
                item.addEventListener('click', () => {
                    const frequencies = JSON.parse(item.dataset.frequencies);
                    selectAsset(item.dataset.symbol, frequencies, item.dataset.defaultFreq);
                    dropdown.classList.remove('active');
                });
            });
        }
        
        dropdown.classList.add('active');
    } catch (error) {
        console.error('Error searching assets:', error);
    }
}

function selectAsset(symbol, frequencies, defaultFreq) {
    SeasonalityState.asset = { 
        symbol, 
        frequencies,
        freq: defaultFreq || frequencies[0] || 'candle_1hour'
    };
    
    document.getElementById('seasonalityAssetSearch').value = '';
    
    const display = document.getElementById('selectedAssetDisplay');
    display.innerHTML = `
        <span class="asset-name">${symbol}</span>
        <span class="clear-asset" onclick="clearSelectedAsset()">&times;</span>
    `;
    display.classList.add('active');
    
    updateFrequencySelector(frequencies, SeasonalityState.asset.freq);
    
    document.getElementById('runAnalysisBtn').disabled = false;
}

function updateFrequencySelector(frequencies, selectedFreq) {
    const container = document.getElementById('frequencySelectorContainer');
    if (!container) return;
    
    const freqLabels = {
        'candle_1min': '1 Min',
        'candle_1hour': '1 Hour',
        'candle_1day': '1 Day'
    };
    
    container.innerHTML = frequencies.map(freq => `
        <button class="freq-btn ${freq === selectedFreq ? 'active' : ''}" data-freq="${freq}">
            ${freqLabels[freq] || freq}
        </button>
    `).join('');
    
    container.querySelectorAll('.freq-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.freq-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            SeasonalityState.asset.freq = btn.dataset.freq;
            
            if (SeasonalityState.analysisData) {
                runAnalysis();
            }
        });
    });
}

window.clearSelectedAsset = function() {
    SeasonalityState.asset = null;
    SeasonalityState.analysisData = null;
    SeasonalityState.customWindow = null;
    SeasonalityState.customWindowEnabled = false;
    
    var display = document.getElementById('selectedAssetDisplay');
    if (display) {
        display.classList.remove('active');
        display.innerHTML = '';
    }
    
    var freqContainer = document.getElementById('frequencySelectorContainer');
    if (freqContainer) {
        freqContainer.innerHTML = '';
    }
    
    var runBtn = document.getElementById('runAnalysisBtn');
    if (runBtn) {
        runBtn.disabled = true;
    }
    
    var toggleSwitch = document.getElementById('customWindowToggle');
    if (toggleSwitch) {
        toggleSwitch.checked = false;
    }
    
    clearCharts();
    clearStats();
    clearDataTable();
};

function initMetricToggle() {
    const buttons = document.querySelectorAll('.metric-btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            SeasonalityState.metric = btn.dataset.metric;
            
            if (SeasonalityState.analysisData) {
                runAnalysis();
            }
        });
    });
}

function initTimeframeSelection() {
    const buttons = document.querySelectorAll('.timeframe-btn');
    
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            SeasonalityState.interval = parseInt(btn.dataset.interval);
            
            if (SeasonalityState.analysisData) {
                runAnalysis();
            }
        });
    });
    
    const toggleSwitch = document.getElementById('customWindowToggle');
    const startInput = document.getElementById('customWindowStart');
    const endInput = document.getElementById('customWindowEnd');
    
    toggleSwitch.addEventListener('change', () => {
        SeasonalityState.customWindowEnabled = toggleSwitch.checked;
        
        if (toggleSwitch.checked) {
            SeasonalityState.customWindow = {
                start: startInput.value,
                end: endInput.value
            };
        } else {
            SeasonalityState.customWindow = null;
        }
        
        if (SeasonalityState.analysisData) {
            runAnalysis();
        }
    });
    
    const handleTimeChange = () => {
        if (SeasonalityState.customWindowEnabled) {
            SeasonalityState.customWindow = {
                start: startInput.value,
                end: endInput.value
            };
            
            if (SeasonalityState.analysisData) {
                runAnalysis();
            }
        }
    };
    
    startInput.addEventListener('change', handleTimeChange);
    endInput.addEventListener('change', handleTimeChange);
}

function initDateRange() {
    const presetButtons = document.querySelectorAll('.date-preset-btn');
    
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            SeasonalityState.dateRange = btn.dataset.range;
            
            const dates = calculateDateRange(btn.dataset.range);
            SeasonalityState.startDate = dates.start;
            SeasonalityState.endDate = dates.end;
            
            document.getElementById('dateRangeStart').value = dates.start;
            document.getElementById('dateRangeEnd').value = dates.end;
            
            if (SeasonalityState.analysisData) {
                runAnalysis();
            }
        });
    });
    
    const dates = calculateDateRange('30y');
    SeasonalityState.startDate = dates.start;
    SeasonalityState.endDate = dates.end;
    document.getElementById('dateRangeStart').value = dates.start;
    document.getElementById('dateRangeEnd').value = dates.end;
    
    document.getElementById('dateRangeStart').addEventListener('change', (e) => {
        SeasonalityState.startDate = e.target.value;
        document.querySelectorAll('.date-preset-btn').forEach(b => b.classList.remove('active'));
        if (SeasonalityState.analysisData) runAnalysis();
    });
    
    document.getElementById('dateRangeEnd').addEventListener('change', (e) => {
        SeasonalityState.endDate = e.target.value;
        document.querySelectorAll('.date-preset-btn').forEach(b => b.classList.remove('active'));
        if (SeasonalityState.analysisData) runAnalysis();
    });
}

function calculateDateRange(range) {
    const end = new Date();
    const start = new Date();
    
    switch (range) {
        case '30y':
            start.setFullYear(end.getFullYear() - 30);
            break;
        case '25y':
            start.setFullYear(end.getFullYear() - 25);
            break;
        case '20y':
            start.setFullYear(end.getFullYear() - 20);
            break;
        case '15y':
            start.setFullYear(end.getFullYear() - 15);
            break;
        case '10y':
            start.setFullYear(end.getFullYear() - 10);
            break;
        case '5y':
            start.setFullYear(end.getFullYear() - 5);
            break;
        case '1y':
            start.setFullYear(end.getFullYear() - 1);
            break;
        case '6m':
            start.setMonth(end.getMonth() - 6);
            break;
        case 'ytd':
            start.setMonth(0, 1);
            break;
    }
    
    return {
        start: start.toISOString().split('T')[0],
        end: end.toISOString().split('T')[0],
    };
}

function initChartTabs() {
    const tabs = document.querySelectorAll('.chart-tab');
    const panels = document.querySelectorAll('.chart-panel');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            
            tab.classList.add('active');
            const panelId = `chart${tab.dataset.chart.charAt(0).toUpperCase() + tab.dataset.chart.slice(1)}`;
            document.getElementById(panelId).classList.add('active');
            
            if (SeasonalityState.analysisData) {
                setTimeout(() => {
                    Object.values(SeasonalityState.charts).forEach(chart => {
                        if (chart) chart.resize();
                    });
                }, 100);
            }
        });
    });
}

function initDataTable() {
    document.getElementById('tablePeriodType').addEventListener('change', (e) => {
        if (SeasonalityState.analysisData) {
            updateDataTable(e.target.value);
        }
    });
}

function initRunAnalysis() {
    document.getElementById('runAnalysisBtn').addEventListener('click', runAnalysis);
}

async function runAnalysis() {
    if (!SeasonalityState.asset) return;
    
    const runBtn = document.getElementById('runAnalysisBtn');
    runBtn.disabled = true;
    runBtn.textContent = 'Analyzing...';
    
    showLoadingOverlay();
    
    const requestBody = {
        symbol: SeasonalityState.asset.symbol,
        freq: SeasonalityState.asset.freq || 'candle_1hour',
        metric: SeasonalityState.metric,
        interval_minutes: SeasonalityState.interval,
        start_date: SeasonalityState.startDate,
        end_date: SeasonalityState.endDate,
    };
    
    if (SeasonalityState.customWindow) {
        requestBody.custom_window = SeasonalityState.customWindow;
    }
    
    if (SeasonalityState.drillDown) {
        requestBody.filter_weekday = SeasonalityState.drillDown.type === 'weekday' ? SeasonalityState.drillDown.value : null;
        requestBody.filter_month = SeasonalityState.drillDown.type === 'month' ? SeasonalityState.drillDown.value : null;
    }
    
    try {
        const response = await fetch('/api/seasonality/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
        });
        
        const responseText = await response.text();
        
        if (!response.ok) {
            let errorMsg = 'Analysis failed';
            try {
                const errorJson = JSON.parse(responseText);
                errorMsg = errorJson.detail || errorMsg;
            } catch (e) {
                errorMsg = responseText || errorMsg;
            }
            throw new Error(errorMsg);
        }
        
        SeasonalityState.analysisData = JSON.parse(responseText);
        
        updateAllCharts();
        updateStats();
        updateDataTable('weekday');
        updateHeatmapYearSelect();
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Analysis failed: ${error.message}`);
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = 'Run Analysis';
        hideLoadingOverlay();
    }
}

function updateAllCharts() {
    const data = SeasonalityState.analysisData;
    if (!data) return;
    
    console.log('updateAllCharts called, data keys:', Object.keys(data));
    console.log('total_returns_index:', data.total_returns_index);
    
    updateTotalReturnsChart(data.total_returns_index);
    updateHeatmapChart(data.calendar);
    updateIntradayChart(data.intraday);
    updatePeriodicCharts(data);
    
    setTimeout(() => {
        Object.values(SeasonalityState.charts).forEach(chart => {
            if (chart) chart.resize();
        });
    }, 200);
}

function updateTotalReturnsChart(totalReturnsData) {
    const container = document.getElementById('totalReturnsChart');
    if (!container || !totalReturnsData) {
        console.log('Total returns: missing container or data');
        return;
    }
    
    const data = totalReturnsData.data || [];
    console.log('Total returns data points:', data.length, data.slice(0, 3));
    
    if (data.length === 0) {
        if (SeasonalityState.charts.totalReturns) {
            SeasonalityState.charts.totalReturns.clear();
        }
        return;
    }
    
    if (!SeasonalityState.charts.totalReturns) {
        SeasonalityState.charts.totalReturns = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.totalReturns;
    
    const times = data.map(d => d.datetime);
    const values = data.map(d => d.value);
    
    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const value = params[0].value;
                const pctChange = ((value - 100) / 100 * 100).toFixed(2);
                const color = value >= 100 ? POSITIVE_COLOR : NEGATIVE_COLOR;
                return `<div style="font-size:12px">
                    <div style="font-weight:bold;margin-bottom:4px">${params[0].axisValue}</div>
                    <div>Index: <span style="color:${color};font-weight:bold">${value.toFixed(2)}</span></div>
                    <div>Change: <span style="color:${color}">${value >= 100 ? '+' : ''}${pctChange}%</span></div>
                </div>`;
            }
        },
        grid: {
            top: 20,
            left: 60,
            right: 30,
            bottom: 50,
        },
        xAxis: {
            type: 'category',
            data: times,
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: 9,
                rotate: 45,
            },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value',
            axisLine: { show: false },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: 10,
            },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        },
        series: [{
            name: 'Total Returns Index',
            type: 'line',
            data: values,
            lineStyle: { width: 1.5, color: '#22D3EE' },
            symbol: 'none',
            areaStyle: {
                color: {
                    type: 'linear',
                    x: 0, y: 0, x2: 0, y2: 1,
                    colorStops: [
                        { offset: 0, color: 'rgba(34, 211, 238, 0.3)' },
                        { offset: 1, color: 'transparent' }
                    ]
                }
            },
            markLine: {
                silent: true,
                lineStyle: { color: '#6b7280', type: 'dashed' },
                data: [
                    { yAxis: 100, label: { position: 'end', formatter: 'Base: 100' } }
                ]
            }
        }]
    };
    
    chart.setOption(option, true);
    setTimeout(() => chart.resize(), 100);
}

function updateHeatmapChart(calendarData) {
    const container = document.getElementById('heatmapChart');
    if (!container || !calendarData) return;
    
    if (!SeasonalityState.charts.heatmap) {
        SeasonalityState.charts.heatmap = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.heatmap;
    const year = calendarData.year || new Date().getFullYear();
    
    const calendarData_formatted = calendarData.data.map(d => [d.date, d.value]);
    
    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            formatter: function(params) {
                const value = params.value[1];
                const color = value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
                return `<div style="font-size:12px">
                    <div>${params.value[0]}</div>
                    <div style="color:${color};font-weight:bold">${value >= 0 ? '+' : ''}${value.toFixed(2)}%</div>
                </div>`;
            }
        },
        visualMap: {
            min: -3,
            max: 3,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            top: 0,
            inRange: {
                color: [NEGATIVE_COLOR, '#1a1a2e', POSITIVE_COLOR]
            },
            textStyle: {
                color: '#a0a0a0'
            }
        },
        calendar: {
            top: 60,
            left: 30,
            right: 30,
            cellSize: ['auto', 15],
            range: year,
            itemStyle: {
                borderWidth: 2,
                borderColor: '#0F172A'
            },
            yearLabel: { show: false },
            monthLabel: {
                color: '#a0a0a0',
                fontSize: 11
            },
            dayLabel: {
                color: '#6b7280',
                fontSize: 10,
                nameMap: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
            },
            splitLine: {
                lineStyle: {
                    color: 'rgba(255,255,255,0.1)'
                }
            }
        },
        series: [{
            type: 'heatmap',
            coordinateSystem: 'calendar',
            data: calendarData_formatted
        }]
    };
    
    chart.setOption(option);
}

function updateIntradayChart(intradayData) {
    const container = document.getElementById('intradayChart');
    if (!container || !intradayData) return;
    
    if (!SeasonalityState.charts.intraday) {
        SeasonalityState.charts.intraday = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.intraday;
    const profile = intradayData.profile || [];
    
    const times = profile.map(d => d.time);
    const values = profile.map(d => d.value);
    const colors = values.map(v => v >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR);
    
    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const d = params[0];
                const color = d.value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR;
                return `<div style="font-size:12px">
                    <div style="font-weight:bold;margin-bottom:4px">${d.axisValue}</div>
                    <div>Avg Return: <span style="color:${color};font-weight:bold">${d.value >= 0 ? '+' : ''}${d.value.toFixed(3)}%</span></div>
                </div>`;
            }
        },
        grid: {
            top: 20,
            left: 50,
            right: 20,
            bottom: 40,
        },
        xAxis: {
            type: 'category',
            data: times,
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: 9,
                rotate: 45,
            },
            splitLine: { show: false },
        },
        yAxis: {
            type: 'value',
            axisLine: { show: false },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: 9,
                formatter: '{value}%'
            },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        },
        series: [{
            name: 'Average',
            type: 'bar',
            data: values.map((v, i) => ({
                value: v,
                itemStyle: { color: colors[i] }
            })),
            barWidth: '60%',
        }]
    };
    
    chart.setOption(option, true);
    chart.resize();
}

function updatePeriodicCharts(data) {
    updateWeekdayChart(data.weekday);
    updateMonthChart(data.month);
    updateDayOfMonthChart(data.day_of_month);
}

function updateWeekdayChart(weekdayData) {
    const container = document.getElementById('weekdayChart');
    if (!container || !weekdayData) return;
    
    if (!SeasonalityState.charts.weekday) {
        SeasonalityState.charts.weekday = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.weekday;
    
    const option = createBarChartOption(weekdayData, 'weekday');
    chart.setOption(option, true);
    chart.resize();
    
    chart.off('click');
    chart.on('click', (params) => {
        const period = weekdayData[params.dataIndex];
        applyDrillDown('weekday', period.period_value, period.period);
    });
}

function updateMonthChart(monthData) {
    const container = document.getElementById('monthChart');
    if (!container || !monthData) return;
    
    if (!SeasonalityState.charts.month) {
        SeasonalityState.charts.month = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.month;
    
    const option = createBarChartOption(monthData, 'month');
    chart.setOption(option, true);
    chart.resize();
    
    chart.off('click');
    chart.on('click', (params) => {
        const period = monthData[params.dataIndex];
        applyDrillDown('month', period.period_value, period.period);
    });
}

function updateDayOfMonthChart(dayData) {
    const container = document.getElementById('dayOfMonthChart');
    if (!container || !dayData) return;
    
    if (!SeasonalityState.charts.dayOfMonth) {
        SeasonalityState.charts.dayOfMonth = echarts.init(container);
    }
    
    const chart = SeasonalityState.charts.dayOfMonth;
    
    const option = createBarChartOption(dayData, 'day_of_month', true);
    chart.setOption(option, true);
    chart.resize();
}

function createBarChartOption(data, type, small = false) {
    const labels = data.map(d => d.period);
    const values = data.map(d => d.value);
    const colors = values.map(v => v >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR);
    
    return {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const d = data[params[0].dataIndex];
                return `<div style="font-size:12px">
                    <div style="font-weight:bold;margin-bottom:4px">${d.period}</div>
                    <div>Avg Return: <span style="color:${d.value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR}">${d.value >= 0 ? '+' : ''}${d.avg_return?.toFixed(3) || d.value.toFixed(3)}%</span></div>
                    <div>Win Rate: ${d.win_rate?.toFixed(1) || '--'}%</div>
                    <div>Count: ${d.count?.toLocaleString() || '--'}</div>
                </div>`;
            }
        },
        grid: {
            top: 10,
            left: small ? 35 : 45,
            right: 10,
            bottom: small ? 20 : 30,
        },
        xAxis: {
            type: 'category',
            data: labels,
            axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: small ? 9 : 10,
                rotate: type === 'day_of_month' ? 45 : 0
            },
        },
        yAxis: {
            type: 'value',
            axisLine: { show: false },
            axisLabel: { 
                color: '#6b7280', 
                fontSize: 9,
                formatter: '{value}%'
            },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        },
        series: [{
            type: 'bar',
            data: values.map((v, i) => ({
                value: v,
                itemStyle: { color: colors[i] }
            })),
            barWidth: small ? '60%' : '50%',
        }]
    };
}

function updateStats() {
    const summary = SeasonalityState.analysisData?.summary;
    if (!summary) return;
    
    const winRateEl = document.getElementById('statWinRate');
    const avgReturnEl = document.getElementById('statAvgReturn');
    const sharpeEl = document.getElementById('statSharpe');
    const maxDDEl = document.getElementById('statMaxDD');
    const volatilityEl = document.getElementById('statVolatility');
    const countEl = document.getElementById('statCount');
    
    winRateEl.textContent = `${summary.win_rate.toFixed(1)}%`;
    winRateEl.className = 'stat-value';
    
    avgReturnEl.textContent = `${summary.avg_return >= 0 ? '+' : ''}${summary.avg_return.toFixed(4)}%`;
    avgReturnEl.className = `stat-value ${summary.avg_return >= 0 ? 'positive' : 'negative'}`;
    
    sharpeEl.textContent = summary.sharpe.toFixed(3);
    sharpeEl.className = `stat-value ${summary.sharpe >= 0 ? 'positive' : 'negative'}`;
    
    maxDDEl.textContent = `${summary.max_drawdown.toFixed(2)}%`;
    maxDDEl.className = 'stat-value negative';
    
    volatilityEl.textContent = `${summary.volatility.toFixed(4)}%`;
    volatilityEl.className = 'stat-value';
    
    countEl.textContent = summary.count.toLocaleString();
    countEl.className = 'stat-value';
}

function updateDataTable(periodType) {
    const tbody = document.getElementById('dataTableBody');
    const data = SeasonalityState.analysisData;
    
    if (!data) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No data available</td></tr>';
        return;
    }
    
    let tableData;
    switch (periodType) {
        case 'weekday':
            tableData = data.weekday;
            break;
        case 'month':
            tableData = data.month;
            break;
        case 'day_of_month':
            tableData = data.day_of_month;
            break;
        default:
            tableData = data.weekday;
    }
    
    if (!tableData || tableData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No data for selected period</td></tr>';
        return;
    }
    
    tbody.innerHTML = tableData.map((row, index) => `
        <tr data-period-type="${periodType}" data-period-value="${row.period_value}" onclick="handleTableRowClick(this)">
            <td>${row.period}</td>
            <td class="${row.avg_return >= 0 ? 'positive' : 'negative'}">
                ${row.avg_return >= 0 ? '+' : ''}${row.avg_return?.toFixed(4) || '--'}%
            </td>
            <td>${row.volatility?.toFixed(4) || '--'}</td>
            <td>${row.rsi?.toFixed(1) || '--'}</td>
            <td>${row.win_rate?.toFixed(1) || '--'}%</td>
            <td>${row.count?.toLocaleString() || '--'}</td>
        </tr>
    `).join('');
}

window.handleTableRowClick = function(row) {
    const periodType = row.dataset.periodType;
    const periodValue = parseInt(row.dataset.periodValue);
    const periodName = row.cells[0].textContent;
    
    if (periodType !== 'day_of_month') {
        applyDrillDown(periodType, periodValue, periodName);
    }
    
    document.querySelectorAll('.data-table tr').forEach(r => r.classList.remove('highlighted'));
    row.classList.add('highlighted');
};

function applyDrillDown(type, value, name) {
    SeasonalityState.drillDown = { type, value, name };
    
    showDrillDownIndicator(type, name);
    
    runAnalysis();
}

function showDrillDownIndicator(type, name) {
    const existing = document.querySelector('.drilldown-indicator');
    if (existing) existing.remove();
    
    const indicator = document.createElement('div');
    indicator.className = 'drilldown-indicator';
    indicator.innerHTML = `
        <span>Filtered by: <strong>${type === 'weekday' ? 'Day: ' : 'Month: '}${name}</strong></span>
        <span class="clear-drilldown" onclick="clearDrillDown()">Clear Filter</span>
    `;
    
    const visualizationArea = document.querySelector('.visualization-area');
    visualizationArea.insertBefore(indicator, visualizationArea.querySelector('.chart-tabs'));
}

window.clearDrillDown = function() {
    SeasonalityState.drillDown = null;
    
    const indicator = document.querySelector('.drilldown-indicator');
    if (indicator) indicator.remove();
    
    document.querySelectorAll('.data-table tr').forEach(r => r.classList.remove('highlighted'));
    
    runAnalysis();
};

function updateHeatmapYearSelect() {
    const select = document.getElementById('heatmapYearSelect');
    const years = SeasonalityState.analysisData?.calendar?.available_years || [];
    
    if (years.length === 0) {
        select.innerHTML = '<option value="">No data</option>';
        return;
    }
    
    select.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join('');
    select.value = Math.max(...years);
    
    select.onchange = () => {
        const year = parseInt(select.value);
        loadCalendarForYear(year);
    };
}

async function loadCalendarForYear(year) {
    if (!SeasonalityState.asset) return;
    
    try {
        const response = await fetch(
            `/api/seasonality/calendar/${encodeURIComponent(SeasonalityState.asset.symbol)}?` +
            `freq=${SeasonalityState.asset.freq}&year=${year}` +
            `&start_date=${SeasonalityState.startDate || ''}` +
            `&end_date=${SeasonalityState.endDate || ''}`
        );
        
        if (response.ok) {
            const calendarData = await response.json();
            updateHeatmapChart(calendarData);
        }
    } catch (error) {
        console.error('Error loading calendar:', error);
    }
}

function showLoadingOverlay() {
    const vizArea = document.querySelector('.visualization-area');
    if (!vizArea.querySelector('.loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = '<div class="loading-spinner"></div>';
        vizArea.appendChild(overlay);
    }
}

function hideLoadingOverlay() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) overlay.remove();
}

function clearCharts() {
    Object.values(SeasonalityState.charts).forEach(chart => {
        if (chart) chart.clear();
    });
}

function clearStats() {
    ['statWinRate', 'statAvgReturn', 'statSharpe', 'statMaxDD', 'statVolatility', 'statCount'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = '--';
            el.className = 'stat-value';
        }
    });
}

function clearDataTable() {
    const tbody = document.getElementById('dataTableBody');
    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-state">Select an asset to view data</td></tr>';
    }
}

function resizeCharts() {
    Object.values(SeasonalityState.charts).forEach(chart => {
        if (chart) chart.resize();
    });
}

window.addEventListener('resize', () => {
    resizeCharts();
});

window.seasonalityState = SeasonalityState;
