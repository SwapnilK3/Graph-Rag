// ── Configuration ──────────────────────────────────────────────────
// When accessed from browser on port 3000, call backend on same host:8080
// Works for both Docker (host-mapped ports) and local dev
const API_BASE_URL = location.port === '3000'
    ? `${location.protocol}//${location.hostname}:8080`
    : '';

// ── DOM Elements ────────────────────────────────────────────────────
const chatHistory = document.getElementById('chat-history');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const welcomeHero = document.getElementById('welcome-hero');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const domainSelect = document.getElementById('domain-select');

// ── State ───────────────────────────────────────────────────────────
let isProcessing = false;
let activeDomain = 'medical';

// ── Event Listeners ─────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Auto-resize textarea
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 200) + 'px';
});

// Domain selector
if (domainSelect) {
    domainSelect.addEventListener('change', (e) => {
        activeDomain = e.target.value;
    });
}

// Sidebar domain buttons
document.querySelectorAll('.domain-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const domain = btn.dataset.domain;
        if (domain) {
            activeDomain = domain;
            if (domainSelect) domainSelect.value = domain;
            // Visual feedback
            document.querySelectorAll('.domain-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }
    });
});

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
        chatHistory.innerHTML = '';
        // Re-create welcome hero
        chatHistory.innerHTML = `
            <div id="welcome-hero" class="max-w-3xl mx-auto flex flex-col items-center justify-center mt-20 space-y-8 animate-fade-in-up">
                <div class="text-center space-y-3">
                    <h1 class="text-4xl md:text-5xl font-bold tracking-tight text-white">Where knowledge begins.</h1>
                    <p class="text-zinc-500 text-lg">Search any knowledge graph with intelligent reasoning</p>
                </div>
                <div class="flex flex-wrap justify-center gap-2 max-w-xl">
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"Common side effects of Aspirin?"</button>
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"What treats headaches?"</button>
                    <button class="suggestion-chip px-4 py-2 bg-white/5 border border-white/5 rounded-full text-sm text-zinc-400 hover:text-white hover:bg-white/10 transition-all">"How is aspirin connected to peptic ulcer?"</button>
                </div>
            </div>
        `;
        // Re-bind suggestion chips
        document.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                queryInput.value = chip.innerText.replace(/"/g, '');
                handleSend();
            });
        });
    });
}

// ── Health Check ────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            const data = await res.json();
            setStatus('connected', `Connected (${data.version || 'v2'})`);
        } else {
            setStatus('error', 'Backend error');
        }
    } catch (e) {
        setStatus('error', 'Disconnected');
        // Retry in 5 seconds
        setTimeout(checkHealth, 5000);
    }
}

function setStatus(state, text) {
    if (statusDot) {
        statusDot.className = 'w-2 h-2 rounded-full ' + {
            'connected': 'bg-emerald-500',
            'error': 'bg-red-500',
            'loading': 'bg-yellow-500',
        }[state];
    }
    if (statusText) statusText.textContent = text;
}

// Run health check on load
checkHealth();

// ── Core Logic ──────────────────────────────────────────────────────
async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    const hero = document.getElementById('welcome-hero');
    if (hero) hero.remove();

    isProcessing = true;
    toggleLoading(true);

    // Append User Message
    appendMessage('user', query);
    queryInput.value = '';
    queryInput.style.height = 'auto';

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, domain: activeDomain })
        });
        
        if (!response.ok) {
            // Backend returned an error status
            let errorMsg = `Backend error (${response.status})`;
            try {
                const errData = await response.json();
                errorMsg = errData.detail || errorMsg;
            } catch (_) {}
            appendMessage('error', errorMsg);
            return;
        }

        const data = await response.json();
        appendMessage('assistant', data);
    } catch (error) {
        console.error('API Error:', error);
        appendMessage('error', 
            `Connection failed: ${error.message}. Check that backend is running on ${API_BASE_URL || 'same origin'}.`
        );
        setStatus('error', 'Disconnected');
    } finally {
        isProcessing = false;
        toggleLoading(false);
    }
}

function appendMessage(role, data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `max-w-4xl mx-auto flex gap-6 animate-fade-in-up ${role === 'user' ? 'justify-end' : ''}`;
    
    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="bg-indigo-600/10 border border-indigo-500/20 px-5 py-3 rounded-2xl text-slate-200">
                ${escapeHtml(data)}
            </div>
            <div class="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0 flex items-center justify-center text-[10px] font-bold">YOU</div>
        `;
    } else if (role === 'error') {
        msgDiv.innerHTML = `<div class="bg-red-500/10 border border-red-500/20 px-4 py-2 rounded-lg text-red-400 text-sm flex items-center gap-2"><i data-lucide="alert-circle" class="w-4 h-4"></i> ${escapeHtml(data)}</div>`;
    } else {
        // Assistant Message
        const answer = data.answer || data.context || '(No answer generated)';
        const context = data.context || '';
        const thought_process = data.thought_process || [];
        const subgraph = data.subgraph || null;
        const intent = data.intent || 'unknown';
        const strategy = data.strategy || 'unknown';
        const hopDepth = data.hop_depth || 0;
        const timing = data.timing || {};
        
        // Out-of-scope styling
        const isOutOfScope = intent === 'out_of_scope';

        let thoughtsHtml = '';
        if (thought_process.length > 0) {
            thoughtsHtml = `
                <div class="mb-6 p-4 rounded-xl bg-white/5 border border-white/5">
                    <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-indigo-400 transition-all uppercase tracking-widest">
                        <i data-lucide="brain" class="w-3.5 h-3.5"></i> Thought Process (${thought_process.length} step${thought_process.length > 1 ? 's' : ''})
                    </button>
                    <div class="mt-4 space-y-4 hidden pt-2">
                        ${thought_process.map(t => {
                            if (t.action === 'domain_guard') {
                                return `<div class="thought-step">
                                    <p class="text-xs font-bold text-amber-400">Domain Guard</p>
                                    <p class="text-sm text-slate-400">Result: <span class="text-amber-300">${t.result}</span></p>
                                    ${t.reason ? `<p class="text-[10px] text-slate-500 italic">${t.reason}</p>` : ''}
                                </div>`;
                            }
                            return `<div class="thought-step">
                                <p class="text-xs font-bold text-indigo-400">Step ${t.step}</p>
                                <p class="text-sm text-slate-400">Searching for: <span class="text-slate-200">${(t.nodes || []).join(', ') || '—'}</span></p>
                                <p class="text-[10px] text-slate-500 italic">Intent: ${t.intent || '—'} | Facts: ${t.found_facts ?? '—'} | Quality: ${t.quality_score ?? '—'}</p>
                            </div>`;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        // Metadata badge
        const metaHtml = `
            <div class="flex flex-wrap items-center gap-2 mb-4">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${isOutOfScope ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' : 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/20'}">${intent}</span>
                <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-white/5 text-zinc-400 border border-white/5">${strategy}</span>
                ${hopDepth > 0 ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-white/5 text-zinc-400 border border-white/5">${hopDepth}-hop</span>` : ''}
                ${timing.domain_guard_ms ? `<span class="text-[10px] text-zinc-600">${Math.round(Object.values(timing).reduce((a,b) => a+b, 0))}ms</span>` : ''}
            </div>
        `;

        const graphId = `graph-${Math.random().toString(36).substr(2, 9)}`;
        
        const hasGraph = subgraph && subgraph.nodes && subgraph.nodes.length > 0;
        const hasContext = context && context.trim().length > 0;

        let bottomPanel = '';
        if (hasGraph || hasContext) {
            bottomPanel = `
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
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="w-8 h-8 rounded-full ${isOutOfScope ? 'bg-amber-600' : 'bg-indigo-600'} flex-shrink-0 flex items-center justify-center text-white"><i data-lucide="${isOutOfScope ? 'shield-off' : 'bot'}" class="w-5 h-5"></i></div>
            <div class="flex-1 space-y-4">
                ${metaHtml}
                ${thoughtsHtml}
                <div class="prose prose-invert max-w-none text-slate-200 leading-relaxed font-light">
                    ${formatAnswer(answer)}
                </div>
                ${bottomPanel}
            </div>
        `;
        
        setTimeout(() => {
            if (hasGraph) renderGraph(subgraph, graphId);
            lucide.createIcons();
        }, 100);
    }
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    lucide.createIcons();
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
    return text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-300">$1</strong>')
               .replace(/\n/g, '<br/>');
}

function toggleLoading(show) {
    sendBtn.innerHTML = show 
        ? `<i data-lucide="loader" class="w-5 h-5 animate-spin"></i>` 
        : `<i data-lucide="arrow-right" class="w-5 h-5"></i>`;
    sendBtn.disabled = show;
    lucide.createIcons();
}

// ── Visualization Engine (D3.js) ──────────────────────────────────
function renderGraph(subgraph, containerId) {
    const container = document.getElementById(containerId);
    if (!container || !subgraph || !subgraph.nodes || subgraph.nodes.length === 0) return;
    
    const width = container.clientWidth || 400;
    const height = 300;
    
    // Clear any existing SVG
    d3.select(`#${containerId}`).selectAll('svg').remove();
    
    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("viewBox", [0, 0, width, height]);

    // Build node/link data with proper cloning
    const nodeMap = new Map();
    const nodes = subgraph.nodes.map(d => {
        const node = { ...d };
        nodeMap.set(d.id, node);
        return node;
    });
    
    // Only include links where both source and target exist
    const links = (subgraph.relationships || [])
        .filter(d => nodeMap.has(d.source_id) && nodeMap.has(d.target_id))
        .map(d => ({
            source: d.source_id,
            target: d.target_id,
            type: d.type
        }));

    if (nodes.length === 0) return;

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-150))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20));

    // Edge labels
    const linkGroup = svg.append("g").attr("class", "links");
    const link = linkGroup.selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke-width", 1);

    const linkLabel = linkGroup.selectAll("text")
        .data(links)
        .join("text")
        .attr("fill", "#555")
        .attr("font-size", "7px")
        .attr("text-anchor", "middle")
        .text(d => d.type);

    const node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", 6)
        .attr("fill", d => getColor(d.label));

    node.append("text")
        .text(d => d.name || d.label)
        .attr("x", 8)
        .attr("y", 4)
        .attr("fill", "#94a3b8")
        .attr("font-size", "10px")
        .attr("font-weight", "bold");

    // Tooltip on hover
    node.append("title")
        .text(d => `${d.label}: ${d.name || '(unnamed)'}`);

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        linkLabel
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);

        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function getColor(label) {
        const colors = {
            'Drug': '#6366f1', 'Disease': '#f43f5e', 'SideEffect': '#eab308',
            'Manufacturer': '#10b981', 'Symptom': '#f97316', 'Gene': '#8b5cf6',
            'Protein': '#ec4899', 'Pathway': '#06b6d4',
        };
        return colors[label] || '#94a3b8';
    }

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
}
