// ── Configuration ──────────────────────────────────────────────────
const API_BASE_URL = location.port === '3000'
    ? `${location.protocol}//${location.hostname}:8080`
    : '';

// ── DOM Elements ────────────────────────────────────────────────────
const chatHistory = document.getElementById('chat-history');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const modeToggle = document.getElementById('mode-toggle');
const activeKgBadge = document.getElementById('active-kg-badge');
const kgListEl = document.getElementById('kg-list');

// Modal elements
const connectModal = document.getElementById('connect-modal');
const openModalBtn = document.getElementById('open-connect-modal');
const closeModalBtn = document.getElementById('close-connect-modal');
const connectBtn = document.getElementById('connect-btn');
const connectStatus = document.getElementById('connect-status');

// ── State ───────────────────────────────────────────────────────────
let isProcessing = false;
let activeKgId = 'default';
let agentMode = false; // false = Fast (/query), true = Agent (/agent/query)
let knownGraphs = []; // cached from GET /graphs

// ── Initialization ──────────────────────────────────────────────────
async function init() {
    await checkHealth();
    await loadGraphs();
    lucide.createIcons();
}

// ── Event Listeners ─────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
});

// Mode toggle (Fast ↔ Agent)
if (modeToggle) {
    modeToggle.addEventListener('click', () => {
        agentMode = !agentMode;
        if (agentMode) {
            modeToggle.textContent = 'Agent';
            modeToggle.className = 'px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider bg-purple-500/10 text-purple-400 border border-purple-500/20 hover:bg-purple-500/20 transition-colors cursor-pointer';
            queryInput.placeholder = 'Ask complex questions... (Agent mode — multi-step reasoning)';
        } else {
            modeToggle.textContent = 'Fast';
            modeToggle.className = 'px-2 py-1 rounded text-[10px] font-bold uppercase tracking-wider bg-brand-500/10 text-brand-400 border border-brand-500/20 hover:bg-brand-500/20 transition-colors cursor-pointer';
            queryInput.placeholder = 'Ask anything...';
        }
    });
}

// Suggestions
document.querySelectorAll('.suggestion-chip').forEach(btn => {
    btn.addEventListener('click', () => {
        queryInput.value = btn.innerText.replace(/"/g, '');
        handleSend();
    });
});

// New thread
const newThreadBtn = document.getElementById('new-thread');
if (newThreadBtn) {
    newThreadBtn.addEventListener('click', () => {
        chatHistory.innerHTML = `
            <div id="welcome-hero"
                class="max-w-3xl mx-auto flex flex-col items-center justify-center mt-20 space-y-8 animate-fade-in-up">
                <div class="text-center space-y-3">
                    <h1 class="text-4xl md:text-5xl font-bold tracking-tight text-white">Where knowledge begins.</h1>
                    <p class="text-zinc-500 text-lg">Connect any Neo4j graph · Auto-discover schema · Query with AI</p>
                </div>
                <div class="flex flex-wrap justify-center gap-2 max-w-xl">
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"Common side effects of Aspirin?"</button>
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"What treats headaches?"</button>
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"How is aspirin connected to peptic ulcer?"</button>
                </div>
            </div>`;
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                queryInput.value = chip.innerText.replace(/"/g, '');
                handleSend();
            });
        });
    });
}

// ── Modal Logic ─────────────────────────────────────────────────────
if (openModalBtn) openModalBtn.addEventListener('click', () => {
    const dbInput = document.getElementById('db-database');
    const defaultGraph = knownGraphs.find(g => g.kg_id === 'default');
    if (dbInput && !dbInput.value && defaultGraph?.database) {
        dbInput.value = defaultGraph.database;
    }
    connectModal.classList.remove('hidden');
    connectStatus.classList.add('hidden');
    lucide.createIcons();
});
if (closeModalBtn) closeModalBtn.addEventListener('click', () => {
    connectModal.classList.add('hidden');
});

connectBtn.addEventListener('click', async () => {
    const uri = document.getElementById('db-uri').value.trim();
    const database = document.getElementById('db-database').value.trim();
    const user = document.getElementById('db-user').value.trim();
    const pass = document.getElementById('db-pass').value.trim();
    const name = document.getElementById('db-name').value.trim();

    if (!uri || !user || !pass) {
        showConnectStatus('error', 'URI, Username, and Password are required.');
        return;
    }

    connectBtn.disabled = true;
    connectBtn.innerHTML = '<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> Discovering schema...';
    lucide.createIcons();
    showConnectStatus('loading', 'Connecting and running schema discovery...');

    try {
        const payload = { uri, username: user, password: pass, name };
        if (database) payload.database = database;

        const res = await fetch(`${API_BASE_URL}/graphs/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (res.ok) {
            const configSummary = data.config_summary || {};
            const initBits = [];
            if (data.pipeline_initialized) initBits.push('pipeline ready');
            if (data.memory_layer_initialized) initBits.push('memory layer ready');

            showConnectStatus('success',
                `Connected! Found ${data.schema_summary?.total_nodes || 0} nodes, ` +
                `${data.schema_summary?.total_relationships || 0} relationships. ` +
                `Labels: ${(data.schema_summary?.node_labels || []).join(', ')}. ` +
                `Intents: ${configSummary.intent_count || 0}. ` +
                `${initBits.length ? initBits.join(' · ') : 'pipeline pending'}`
            );

            // Reload sidebar, auto-select new graph
            await loadGraphs();
            selectKg(data.kg_id);

            // Close modal after brief delay to show success
            setTimeout(() => {
                connectModal.classList.add('hidden');
                // Clear fields
                document.getElementById('db-uri').value = '';
                document.getElementById('db-database').value = '';
                document.getElementById('db-user').value = '';
                document.getElementById('db-pass').value = '';
                document.getElementById('db-name').value = '';
            }, 1500);
        } else {
            showConnectStatus('error', data.detail || 'Connection failed.');
        }
    } catch (e) {
        showConnectStatus('error', `Network error: ${e.message}`);
    } finally {
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i> Connect & Discover';
        lucide.createIcons();
    }
});

function showConnectStatus(type, msg) {
    connectStatus.classList.remove('hidden');
    const colors = {
        error: 'bg-red-500/10 border border-red-500/20 text-red-400',
        success: 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400',
        loading: 'bg-blue-500/10 border border-blue-500/20 text-blue-400',
    };
    connectStatus.className = `px-4 py-3 rounded-lg text-xs ${colors[type] || colors.error}`;
    connectStatus.textContent = msg;
}

// ── Graph Management ────────────────────────────────────────────────
async function loadGraphs() {
    try {
        const res = await fetch(`${API_BASE_URL}/graphs`);
        if (!res.ok) return;
        const data = await res.json();
        knownGraphs = data.graphs || [];
        renderKgList();
    } catch (e) {
        console.warn('Failed to load graphs:', e);
    }
}

function renderKgList() {
    if (!kgListEl) return;

    if (knownGraphs.length === 0) {
        kgListEl.innerHTML = '<div class="px-3 py-2 text-xs text-zinc-600 italic">No graphs connected yet</div>';
        return;
    }

    kgListEl.innerHTML = knownGraphs.map(g => {
        const isActive = g.kg_id === activeKgId;
        const isDefault = g.kg_id === 'default';
        const nodeCount = g.total_nodes || 0;
        const labels = (g.node_labels || []).slice(0, 3).join(', ');

        return `
            <div class="group flex items-center justify-between px-3 py-2 rounded-lg text-sm font-medium transition-colors cursor-pointer
                        ${isActive ? 'bg-white/8 text-white' : 'text-zinc-400 hover:text-white hover:bg-white/5'}"
                 onclick="selectKg('${g.kg_id}')">
                <div class="flex items-center space-x-3 overflow-hidden min-w-0">
                    <span class="w-2 h-2 rounded-full flex-shrink-0 ${isDefault ? 'bg-blue-500' : 'bg-brand-500'}"></span>
                    <div class="min-w-0">
                        <div class="text-sm truncate">${escapeHtml(g.name)}</div>
                        <div class="text-[10px] text-zinc-600">${nodeCount} nodes${labels ? ' · ' + labels : ''}</div>
                    </div>
                </div>
                ${!isDefault ? `
                    <button onclick="event.stopPropagation(); deleteKg('${g.kg_id}')"
                            class="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition-all flex-shrink-0"
                            title="Remove this graph">
                        <i data-lucide="trash-2" class="w-3 h-3"></i>
                    </button>
                ` : ''}
            </div>`;
    }).join('');

    lucide.createIcons();
}

window.selectKg = function(kgId) {
    activeKgId = kgId;
    if (activeKgBadge) {
        const g = knownGraphs.find(g => g.kg_id === kgId);
        activeKgBadge.textContent = g ? g.name : kgId;
    }
    renderKgList();
};

window.deleteKg = async function(kgId) {
    if (!confirm('Remove this knowledge graph? Credentials and cached schema will be deleted.')) return;

    try {
        const res = await fetch(`${API_BASE_URL}/graphs/${kgId}`, { method: 'DELETE' });
        if (res.ok) {
            // If we deleted the active graph, switch to default
            if (activeKgId === kgId) {
                selectKg('default');
            }
            await loadGraphs();
        } else {
            const data = await res.json();
            alert(data.detail || 'Failed to delete');
        }
    } catch (e) {
        alert(`Delete failed: ${e.message}`);
    }
};

// ── Health Check ────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            const data = await res.json();
            setStatus('connected', `Connected (${data.version || 'v3'})`);
        } else {
            setStatus('error', 'Backend error');
        }
    } catch (e) {
        setStatus('error', 'Disconnected');
        setTimeout(checkHealth, 5000);
    }
}

function setStatus(state, text) {
    if (statusDot) {
        statusDot.className = 'w-2 h-2 rounded-full ' + ({
            'connected': 'bg-emerald-500', 'error': 'bg-red-500', 'loading': 'bg-yellow-500',
        })[state];
    }
    if (statusText) statusText.textContent = text;
}

// ── Core Logic ──────────────────────────────────────────────────────
async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    const hero = document.getElementById('welcome-hero');
    if (hero) hero.remove();

    isProcessing = true;
    toggleLoading(true);
    appendMessage('user', query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    const endpoint = agentMode ? '/agent/query' : '/query';

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, kg_id: activeKgId })
        });

        if (!response.ok) {
            let errorMsg = `Backend error (${response.status})`;
            try {
                const errData = await response.json();
                errorMsg = errData.detail || errorMsg;
            } catch (_) { }
            if (errorMsg.length > 200) {
                errorMsg = errorMsg.substring(0, 200) + '…';
            }
            appendMessage('error', errorMsg);
            return;
        }

        const data = await response.json();

        if (agentMode) {
            appendAgentMessage(data);
        } else {
            appendMessage('assistant', data);
        }
    } catch (error) {
        console.error('API Error:', error);
        appendMessage('error',
            `Connection failed: ${error.message}. Ensure backend is running.`
        );
        setStatus('error', 'Disconnected');
    } finally {
        isProcessing = false;
        toggleLoading(false);
    }
}

// ── V2.5 Pipeline Response ──────────────────────────────────────────
function appendMessage(role, data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `max-w-4xl mx-auto flex gap-6 animate-fade-in-up ${role === 'user' ? 'justify-end' : ''}`;

    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="bg-indigo-600/10 border border-indigo-500/20 px-5 py-3 rounded-2xl text-slate-200">${escapeHtml(data)}</div>
            <div class="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0 flex items-center justify-center text-[10px] font-bold">YOU</div>`;
    } else if (role === 'error') {
        msgDiv.innerHTML = `<div class="bg-red-500/10 border border-red-500/20 px-4 py-3 rounded-lg text-red-400 text-sm flex items-center gap-2"><i data-lucide="alert-circle" class="w-4 h-4 flex-shrink-0"></i><span>${escapeHtml(data)}</span></div>`;
    } else {
        const answer = data.answer || data.context || '(No answer generated)';
        const context = data.context || '';
        const thought_process = data.thought_process || [];
        const subgraph = data.subgraph || null;
        const intent = data.intent || 'unknown';
        const strategy = data.strategy || 'unknown';
        const hopDepth = data.hop_depth || 0;
        const timing = data.timing || {};
        const isOutOfScope = intent === 'out_of_scope';

        const thoughtsHtml = buildThoughtsPanel(thought_process);
        const metaHtml = buildMetaBadges(intent, strategy, hopDepth, timing, isOutOfScope, false);
        const graphId = `graph-${Math.random().toString(36).substr(2, 9)}`;
        const hasGraph = subgraph && subgraph.nodes && subgraph.nodes.length > 0;
        const hasContext = context && context.trim().length > 0;
        const bottomPanel = buildBottomPanel(hasGraph, hasContext, subgraph, context, graphId);

        msgDiv.innerHTML = `
            <div class="w-8 h-8 rounded-full ${isOutOfScope ? 'bg-amber-600' : 'bg-indigo-600'} flex-shrink-0 flex items-center justify-center text-white"><i data-lucide="${isOutOfScope ? 'shield-off' : 'bot'}" class="w-5 h-5"></i></div>
            <div class="flex-1 space-y-4">
                ${metaHtml}
                ${thoughtsHtml}
                <div class="prose prose-invert max-w-none text-slate-200 leading-relaxed font-light">${formatAnswer(answer)}</div>
                ${bottomPanel}
            </div>`;

        setTimeout(() => {
            if (hasGraph) renderGraph(subgraph, graphId);
            lucide.createIcons();
        }, 100);
    }

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    lucide.createIcons();
}

// ── V3 Agent Response ───────────────────────────────────────────────
function appendAgentMessage(data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'max-w-4xl mx-auto flex gap-6 animate-fade-in-up';

    const answer = data.answer || '(No answer generated)';
    const planHistory = data.plan_history || [];
    const subResults = data.sub_results || [];
    const memoryHit = data.memory_hit || false;
    const iterations = data.iterations || 0;
    const timing = data.timing || {};

    // Build agent plan trace
    let planHtml = '';
    if (planHistory.length > 0 || memoryHit) {
        const planSteps = planHistory.map(p => `
            <div class="thought-step">
                <p class="text-xs font-bold text-purple-400">Iteration ${p.iteration + 1}</p>
                <div class="space-y-1">
                    ${(p.sub_questions || []).map(sq => `<p class="text-sm text-slate-400">→ "${escapeHtml(sq)}"</p>`).join('')}
                </div>
                ${p.gaps ? `<p class="text-[10px] text-amber-400 italic">Gaps: ${p.gaps.join(', ')}</p>` : ''}
            </div>
        `).join('');
        planHtml = `
            <div class="mb-6 p-4 rounded-xl bg-white/5 border border-white/5">
                <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-purple-400 transition-all uppercase tracking-widest">
                    <i data-lucide="workflow" class="w-3.5 h-3.5"></i> Agent Plan (${iterations} iteration${iterations !== 1 ? 's' : ''})${memoryHit ? ' • Memory Hit ⚡' : ''}
                </button>
                <div class="mt-4 space-y-4 hidden pt-2">
                    ${memoryHit ? '<div class="thought-step"><p class="text-xs font-bold text-emerald-400">⚡ Memory Recall</p><p class="text-sm text-slate-400">Found matching answer from past queries. Returning cached result.</p></div>' : ''}
                    ${planSteps}
                </div>
            </div>`;
    }

    // Aggregate subgraph from all sub-results
    let mergedNodes = [], mergedRels = [];
    let allContexts = [];
    for (const sr of subResults) {
        const sg = sr.subgraph || {};
        mergedNodes.push(...(sg.nodes || []));
        mergedRels.push(...(sg.relationships || []));
        if (sr.context && sr.context.trim()) allContexts.push(sr.context);
    }
    // Deduplicate nodes by id
    const seenIds = new Set();
    mergedNodes = mergedNodes.filter(n => { if (seenIds.has(n.id)) return false; seenIds.add(n.id); return true; });
    const mergedSubgraph = { nodes: mergedNodes, relationships: mergedRels };
    const mergedContext = allContexts.join('\n---\n');

    const graphId = `agent-graph-${Math.random().toString(36).substr(2, 9)}`;
    const hasGraph = mergedNodes.length > 0;
    const hasContext = mergedContext.trim().length > 0;

    // Timing summary
    const totalMs = Object.values(timing).reduce((a, b) => a + b, 0);

    // Meta badges
    const metaHtml = `
        <div class="flex flex-wrap items-center gap-2 mb-4">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-purple-500/10 text-purple-400 border border-purple-500/20">agent</span>
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-white/5 text-zinc-400 border border-white/5">${iterations} iteration${iterations !== 1 ? 's' : ''}</span>
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-white/5 text-zinc-400 border border-white/5">${subResults.length} sub-queries</span>
            ${memoryHit ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">memory hit</span>' : ''}
            ${totalMs ? `<span class="text-[10px] text-zinc-600">${Math.round(totalMs)}ms</span>` : ''}
        </div>`;

    const bottomPanel = buildBottomPanel(hasGraph, hasContext, mergedSubgraph, mergedContext, graphId);

    msgDiv.innerHTML = `
        <div class="w-8 h-8 rounded-full bg-purple-600 flex-shrink-0 flex items-center justify-center text-white"><i data-lucide="sparkles" class="w-5 h-5"></i></div>
        <div class="flex-1 space-y-4">
            ${metaHtml}
            ${planHtml}
            <div class="prose prose-invert max-w-none text-slate-200 leading-relaxed font-light">${formatAnswer(answer)}</div>
            ${bottomPanel}
        </div>`;

    setTimeout(() => {
        if (hasGraph) renderGraph(mergedSubgraph, graphId);
        lucide.createIcons();
    }, 100);

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    lucide.createIcons();
}

// ── Shared UI Builders ──────────────────────────────────────────────
function buildThoughtsPanel(thought_process) {
    if (!thought_process || thought_process.length === 0) return '';
    return `
        <div class="mb-6 p-4 rounded-xl bg-white/5 border border-white/5">
            <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-indigo-400 transition-all uppercase tracking-widest">
                <i data-lucide="brain" class="w-3.5 h-3.5"></i> Thought Process (${thought_process.length} step${thought_process.length > 1 ? 's' : ''})
            </button>
            <div class="mt-4 space-y-4 hidden pt-2">
                ${thought_process.map(t => {
        if (t.action === 'domain_guard') {
            return `<div class="thought-step"><p class="text-xs font-bold text-amber-400">Domain Guard</p><p class="text-sm text-slate-400">Result: <span class="text-amber-300">${t.result}</span></p>${t.reason ? `<p class="text-[10px] text-slate-500 italic">${t.reason}</p>` : ''}</div>`;
        }
        return `<div class="thought-step"><p class="text-xs font-bold text-indigo-400">Step ${t.step}</p><p class="text-sm text-slate-400">Searching: <span class="text-slate-200">${(t.nodes || []).join(', ') || '—'}</span></p><p class="text-[10px] text-slate-500 italic">Intent: ${t.intent || '—'} | Facts: ${t.found_facts ?? '—'} | Quality: ${t.quality_score ?? '—'}</p></div>`;
    }).join('')}
            </div>
        </div>`;
}

function buildMetaBadges(intent, strategy, hopDepth, timing, isOutOfScope, isAgent) {
    const totalMs = Object.values(timing).reduce((a, b) => a + b, 0);
    return `
        <div class="flex flex-wrap items-center gap-2 mb-4">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${isOutOfScope ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'}">${intent}</span>
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-white/5 text-zinc-400 border border-white/5">${strategy}</span>
            ${hopDepth > 0 ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-white/5 text-zinc-400 border border-white/5">${hopDepth}-hop</span>` : ''}
            ${totalMs ? `<span class="text-[10px] text-zinc-600">${Math.round(totalMs)}ms</span>` : ''}
        </div>`;
}

function buildBottomPanel(hasGraph, hasContext, subgraph, context, graphId) {
    if (!hasGraph && !hasContext) return '';
    return `
        <div class="grid grid-cols-1 ${hasGraph && hasContext ? 'md:grid-cols-2' : ''} gap-4 pt-4 border-t border-white/5">
            ${hasGraph ? `
            <div class="space-y-3">
                <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Source Subgraph (${subgraph.nodes.length} nodes, ${(subgraph.relationships || []).length} edges)</p>
                <div id="${graphId}" class="graph-container"></div>
            </div>` : ''}
            ${hasContext ? `
            <div class="space-y-3">
                <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Knowledge Context</p>
                <div class="p-4 rounded-xl bg-white/5 border border-white/5 text-xs text-slate-400 font-mono h-[300px] overflow-y-auto scrollbar-hide whitespace-pre-wrap">${escapeHtml(context)}</div>
            </div>` : ''}
        </div>`;
}

// ── Utilities ───────────────────────────────────────────────────────
function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnswer(text) {
    if (!text || typeof text !== 'string') return '<em class="text-zinc-500">(No answer generated)</em>';
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-300">$1</strong>')
        .replace(/⚠️/g, '<span class="text-amber-400">⚠️</span>')
        .replace(/\n/g, '<br/>');
}

function toggleLoading(show) {
    sendBtn.innerHTML = show
        ? `<i data-lucide="loader" class="w-5 h-5 animate-spin"></i>`
        : `<i data-lucide="arrow-right" class="w-5 h-5"></i>`;
    sendBtn.disabled = show;
    lucide.createIcons();
}

// ── D3 Visualization ────────────────────────────────────────────────
function renderGraph(subgraph, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !subgraph || !subgraph.nodes || subgraph.nodes.length === 0) return;

    const width = container.clientWidth || 400;
    const height = 300;

    d3.select(`#${containerId}`).selectAll('svg').remove();

    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("viewBox", [0, 0, width, height]);

    const nodeMap = new Map();
    const nodes = subgraph.nodes.map(d => {
        const node = { ...d };
        nodeMap.set(d.id, node);
        return node;
    });

    const links = (subgraph.relationships || [])
        .filter(d => nodeMap.has(d.source_id) && nodeMap.has(d.target_id))
        .map(d => ({ source: d.source_id, target: d.target_id, type: d.type }));

    if (nodes.length === 0) return;

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-150))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20));

    const linkGroup = svg.append("g").attr("class", "links");
    const link = linkGroup.selectAll("line").data(links).join("line")
        .attr("class", "link").attr("stroke-width", 1);

    const linkLabel = linkGroup.selectAll("text").data(links).join("text")
        .attr("fill", "#555").attr("font-size", "7px").attr("text-anchor", "middle")
        .text(d => d.type);

    const node = svg.append("g").attr("class", "nodes")
        .selectAll("g").data(nodes).join("g")
        .call(d3.drag()
            .on("start", (e) => { if (!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; })
            .on("drag", (e) => { e.subject.fx = e.x; e.subject.fy = e.y; })
            .on("end", (e) => { if (!e.active) simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; })
        );

    node.append("circle").attr("r", 6).attr("fill", d => getColor(d.label));
    node.append("text").text(d => d.name || d.label).attr("x", 8).attr("y", 4)
        .attr("fill", "#94a3b8").attr("font-size", "10px").attr("font-weight", "bold");
    node.append("title").text(d => `${d.label}: ${d.name || '(unnamed)'}`);

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);
        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function getColor(label) {
        const c = {
            Drug: '#6366f1', Disease: '#f43f5e', SideEffect: '#eab308', Manufacturer: '#10b981',
            Symptom: '#f97316', Gene: '#8b5cf6', Protein: '#ec4899', Pathway: '#06b6d4',
            Therapy: '#14b8a6', Reaction: '#f59e0b', Case: '#6b7280'
        };
        return c[label] || '#94a3b8';
    }
}

// Kick off
init();
