/**
 * Sketch2TikZ Web UI — Frontend Controller
 */

// ── State ───────────────────────────────────────
let currentTaskId = null;
let selectedTaskId = null;   // 用户选择保留的结果版本（refine 的基线）
let currentImageFile = null;
let lastGenFileKey = null;   // 记录上次成功生成时用的图片标识
let abortController = null;
const taskResultCache = new Map();  // task_id -> result 缓存

function getFileKey(file) {
    return file ? `${file.name}-${file.size}-${file.lastModified}` : null;
}

// ── DOM refs ────────────────────────────────────
const $ = (id) => document.getElementById(id);

const uploadZone = $('upload-zone');
const fileInput = $('file-input');
const uploadPreview = $('upload-preview');
const previewImg = $('preview-img');
const btnRemoveFile = $('btn-remove-file');
const customPrompt = $('custom-prompt');
const btnGenerate = $('btn-generate');
const btnText = btnGenerate.querySelector('.btn-text');
const btnSpin = btnGenerate.querySelector('.btn-icon-spin');

const progressArea = $('progress-area');
const progressLabel = $('progress-label');
const progressPercent = $('progress-percent');
const progressFill = $('progress-fill');
const progressMessage = $('progress-message');

const emptyState = $('empty-state');
const resultView = $('result-view');
const resultImg = $('result-img');
const scoreBar = $('score-bar');
const scoreValue = $('score-value');
const scoreDiagnosis = $('score-diagnosis');
const scoreTime = $('score-time');
const codeEditor = $('code-editor');

const btnDlPdf = $('btn-dl-pdf');
const btnDlPng = $('btn-dl-png');
const btnCopyCode = $('btn-copy-code');
const btnDlTex = $('btn-dl-tex');
const comparisonCard = $('comparison-card');
const compImgA = $('comp-img-a');
const compImgB = $('comp-img-b');
const compScoreA = $('comp-score-a');
const compScoreB = $('comp-score-b');
const btnSelectA = $('btn-select-a');
const btnSelectB = $('btn-select-b');

const btnHistory = $('btn-history');
const historyPanel = $('history-panel');
const btnCloseHistory = $('btn-close-history');
const historyList = $('history-list');

const toast = $('toast');
const toastMsg = $('toast-msg');

// ── Init ────────────────────────────────────────
function init() {
    // Upload interactions
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length) handleFileSelect(files[0]);
    });
    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Generate
    btnGenerate.addEventListener('click', startGeneration);

    // Downloads
    btnDlPdf.addEventListener('click', () => downloadFile('pdf'));
    btnDlPng.addEventListener('click', () => downloadFile('png'));
    btnDlTex.addEventListener('click', () => downloadFile('tex'));
    btnCopyCode.addEventListener('click', copyCode);

    // History panel
    btnHistory.addEventListener('click', () => historyPanel.classList.remove('hidden'));
    btnCloseHistory.addEventListener('click', () => historyPanel.classList.add('hidden'));

    // Load history
    loadHistory();
}

// ── File Handling ───────────────────────────────
function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请上传图片文件');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showToast('文件过大，最大支持 10MB');
        return;
    }
    currentImageFile = file;
    selectedTaskId = null;          // 换了图，重置选择状态
    lastGenFileKey = null;
    comparisonCard.classList.add('hidden');
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    uploadPreview.classList.remove('hidden');
    uploadZone.querySelector('.upload-placeholder').classList.add('hidden');
    btnGenerate.disabled = false;
    updateButtonLabel();
}

function clearFile() {
    currentImageFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadPreview.classList.add('hidden');
    uploadZone.querySelector('.upload-placeholder').classList.remove('hidden');
    btnGenerate.disabled = true;
    updateButtonLabel();
}

// ── Generation / Refine ─────────────────────────
async function startGeneration() {
    if (!currentImageFile) return;

    const promptVal = customPrompt.value.trim();
    const currentKey = getFileKey(currentImageFile);
    const canRefine = selectedTaskId && currentKey === lastGenFileKey;

    setGenerating(true);
    resetProgress();
    progressArea.classList.remove('hidden');
    emptyState.classList.add('hidden');
    resultView.classList.add('hidden');
    scoreBar.classList.add('hidden');
    comparisonCard.classList.add('hidden');

    if (canRefine) {
        const form = new FormData();
        form.append('prev_task_id', selectedTaskId);
        form.append('custom_prompt', promptVal);
        // difficulty parameter removed — universal prompt for all
        try {
            const res = await fetch('/api/refine', { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const { task_id } = await res.json();
            currentTaskId = task_id;
            connectSSE(task_id);
        } catch (e) {
            showToast('重新生成失败: ' + e.message);
            setGenerating(false);
            progressLabel.textContent = '重新生成失败';
            progressPercent.textContent = '!';
            progressFill.style.width = '100%';
            progressFill.style.background = 'var(--error)';
        }
    } else {
        const form = new FormData();
        form.append('file', currentImageFile);
        if (promptVal) form.append('custom_prompt', promptVal);
        // difficulty parameter removed — universal prompt for all
        try {
            const res = await fetch('/api/generate', { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const { task_id } = await res.json();
            currentTaskId = task_id;
            selectedTaskId = task_id;
            lastGenFileKey = getFileKey(currentImageFile);
            connectSSE(task_id);
        } catch (e) {
            showToast('生成失败: ' + e.message);
            setGenerating(false);
            progressLabel.textContent = '生成失败';
            progressPercent.textContent = '!';
            progressFill.style.width = '100%';
            progressFill.style.background = 'var(--error)';
        }
    }
}

function connectSSE(taskId) {
    if (abortController) abortController.abort();
    abortController = new AbortController();

    const evtSrc = new EventSource(`/api/tasks/${taskId}/stream`);

    evtSrc.onmessage = (e) => {
        const data = JSON.parse(e.data);
        handleEvent(data);
        if (data.stage === 'close') {
            evtSrc.close();
        }
    };

    evtSrc.onerror = () => {
        evtSrc.close();
    };
}

function handleEvent(ev) {
    const { stage, status, message, data } = ev;

    // Progress bar mapping
    const progMap = {
        vision: { pct: 15, label: '图像分析' },
        codegen: { pct: 40, label: '代码生成' },
        compile: { pct: 65, label: '编译渲染' },
        critic: { pct: 85, label: '质量评审' },
        done: { pct: 100, label: '完成' },
        error: { pct: 100, label: '出错' },
    };
    const info = progMap[stage] || { pct: 0, label: stage };

    progressLabel.textContent = info.label;
    progressPercent.textContent = `${info.pct}%`;
    progressFill.style.width = `${info.pct}%`;
    progressMessage.textContent = message;

    if (status === 'fail' || status === 'retry') {
        progressFill.style.background = status === 'fail' ? 'var(--error)' : 'var(--warning)';
    } else {
        progressFill.style.background = 'linear-gradient(90deg, var(--accent), var(--accent-hover))';
    }

    if (stage === 'done') {
        setGenerating(false);
        showResult(data);
        loadHistory();
    } else if (stage === 'error') {
        setGenerating(false);
        progressLabel.textContent = '生成失败';
        progressPercent.textContent = '!';
        showToast('生成失败: ' + message);
    }
}

function resetProgress() {
    progressLabel.textContent = '准备中...';
    progressPercent.textContent = '0%';
    progressFill.style.width = '0%';
    progressFill.style.background = 'linear-gradient(90deg, var(--accent), var(--accent-hover))';
    progressMessage.textContent = '';
}

function setGenerating(isGen) {
    btnGenerate.disabled = isGen || !currentImageFile;
    btnText.textContent = isGen ? '生成中...' : getGenerateLabel();
    btnSpin.classList.toggle('hidden', !isGen);
}

function getGenerateLabel() {
    const currentKey = getFileKey(currentImageFile);
    const canRefine = selectedTaskId && currentKey === lastGenFileKey;
    return canRefine ? '重新生成' : '开始生成';
}

function updateButtonLabel() {
    if (!btnSpin.classList.contains('hidden')) return; // 生成中不更新
    btnText.textContent = getGenerateLabel();
}

// ── Result Display ──────────────────────────────
function showResult(result) {
    // Cache result for comparison
    if (result && result.task_id) {
        taskResultCache.set(result.task_id, result);
    }

    resultView.classList.remove('hidden');

    if (result.compile_ok) {
        // Load rendered PNG preview
        resultImg.src = `/api/tasks/${result.task_id}/preview?t=${Date.now()}`;

        // Score
        scoreBar.classList.remove('hidden');
        scoreValue.textContent = (result.critic_final_score || result.critic_first_score || 0).toFixed(1) + ' / 5.0';
        scoreDiagnosis.textContent = result.diagnosis || '无诊断信息';
        scoreTime.textContent = (result.total_time || 0) + 's';

        // Code
        codeEditor.value = result.tikz_code || '';
    } else {
        scoreBar.classList.remove('hidden');
        scoreValue.textContent = '—';
        scoreValue.style.color = 'var(--error)';
        scoreDiagnosis.textContent = result.diagnosis || '编译失败';
        scoreTime.textContent = (result.total_time || 0) + 's';
        codeEditor.value = result.tikz_code || '';
        resultImg.src = `/api/tasks/${result.task_id}/preview?t=${Date.now()}`;
    }

    // 如果有上一轮选中的结果且不是同一个 task，展示对比
    if (selectedTaskId && selectedTaskId !== result.task_id) {
        showComparison(selectedTaskId, result.task_id);
    } else {
        comparisonCard.classList.add('hidden');
    }
    updateButtonLabel();
}

// ── Comparison ──────────────────────────────────
function showComparison(baseTaskId, newTaskId) {
    const baseResult = taskResultCache.get(baseTaskId);
    const newResult = taskResultCache.get(newTaskId);
    if (!baseResult || !newResult) return;

    compImgA.src = `/api/tasks/${baseTaskId}/preview?t=${Date.now()}`;
    compImgB.src = `/api/tasks/${newTaskId}/preview?t=${Date.now()}`;

    const baseScore = (baseResult.critic_final_score || baseResult.critic_first_score || 0).toFixed(1);
    const newScore = (newResult.critic_final_score || newResult.critic_first_score || 0).toFixed(1);
    compScoreA.textContent = `评分: ${baseScore} / 5.0`;
    compScoreB.textContent = `评分: ${newScore} / 5.0`;

    btnSelectA.onclick = () => selectVersion(baseTaskId);
    btnSelectB.onclick = () => selectVersion(newTaskId);

    comparisonCard.classList.remove('hidden');
    comparisonCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function selectVersion(taskId) {
    selectedTaskId = taskId;
    currentTaskId = taskId;
    comparisonCard.classList.add('hidden');

    // 更新 UI 为选中的版本
    const result = taskResultCache.get(taskId);
    if (result) {
        showResult(result);
        showToast('已保留选中的版本');
        updateButtonLabel();
    } else {
        // 从 API 重新加载
        try {
            const res = await fetch(`/api/tasks/${taskId}`);
            const task = await res.json();
            if (task.result) {
                taskResultCache.set(taskId, task.result);
                showResult(task.result);
                showToast('已保留选中的版本');
                updateButtonLabel();
            }
        } catch {
            showToast('加载选中版本失败');
        }
    }
}


// ── Downloads ───────────────────────────────────
function downloadFile(fmt) {
    if (!currentTaskId) return;
    const a = document.createElement('a');
    a.href = `/api/download/${currentTaskId}/${fmt}`;
    a.download = `${currentTaskId}_${fmt}`;
    a.click();
}

async function copyCode() {
    const code = codeEditor.value;
    if (!code) return;
    try {
        await navigator.clipboard.writeText(code);
        showToast('代码已复制到剪贴板');
    } catch {
        showToast('复制失败，请手动复制');
    }
}

// ── History ─────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/api/history');
        const items = await res.json();
        renderHistory(items);
    } catch {
        // silent fail
    }
}

function renderHistory(items) {
    if (!items.length) {
        historyList.innerHTML = '<div class="history-empty">暂无历史记录</div>';
        return;
    }
    // Sort by time descending
    items.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    historyList.innerHTML = items.map(item => {
        const date = item.created_at ? new Date(item.created_at * 1000).toLocaleString('zh-CN', { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit' }) : '';
        const scoreText = item.score ? `${item.score.toFixed(1)}/5.0` : (item.compile_ok ? '编译通过' : '失败');
        const thumb = `/api/tasks/${item.task_id}/preview`;
        return `
            <div class="history-item" data-task-id="${item.task_id}">
                <img src="${thumb}" loading="lazy" onerror="this.style.opacity=0.3">
                <div class="history-item-info">
                    <div class="time">${date}</div>
                    <div class="score">${scoreText}${item.custom_prompt ? ' · 自定义' : ''}</div>
                </div>
            </div>
        `;
    }).join('');

    // Click to restore
    historyList.querySelectorAll('.history-item').forEach(el => {
        el.addEventListener('click', () => restoreTask(el.dataset.taskId));
    });
}

async function restoreTask(taskId) {
    try {
        const res = await fetch(`/api/tasks/${taskId}`);
        const task = await res.json();
        if (!task.result) {
            showToast('该任务暂无结果');
            return;
        }
        currentTaskId = taskId;
        selectedTaskId = taskId;
        taskResultCache.set(taskId, task.result);
        showResult(task.result);
        updateButtonLabel();
        emptyState.classList.add('hidden');
        historyPanel.classList.add('hidden');
    } catch {
        showToast('加载历史记录失败');
    }
}

// ── Toast ───────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
    toastMsg.textContent = msg;
    toast.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add('hidden'), 3000);
}

// ── Boot ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
