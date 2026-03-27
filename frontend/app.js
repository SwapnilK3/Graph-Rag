const API_BASE_URL = location.port === '3000' ? `http://${location.hostname}:8080` : '';

const chatHistory = document.getElementById('chat-history');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');
const welcomeHero = document.getElementById('welcome-hero');

// ── State ───────────────────────────────────────────────────────────
let isProcessing = false;

// ── Event Listeners ─────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// Suggestions
document.querySelectorAll('.suggestion-chip').forEach(btn => {
    btn.addEventListener('click', () => {
        queryInput.value = btn.innerText.replace(/"/g, '');
        handleSend();
    });
});

// ── Core Logic ──────────────────────────────────────────────────────
async function handleSend() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    if (welcomeHero) welcomeHero.remove();
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
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        appendMessage('assistant', data);
    } catch (error) {
        console.error('API Error:', error);
        appendMessage('error', 'Connection to V2 Intelligent Layer failed. Ensure backend is running on 8080.');
    } finally {
        isProcessing = false;
        toggleLoading(false);
    }
}

function appendMessage(role, data) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `max-w-4xl mx-auto flex gap-6 ${role === 'user' ? 'justify-end' : ''}`;
    
    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="bg-indigo-600/10 border border-indigo-500/20 px-5 py-3 rounded-2xl text-slate-200">
                ${data}
            </div>
            <div class="w-8 h-8 rounded-full bg-slate-700 flex-shrink-0 flex items-center justify-center text-[10px] font-bold">YOU</div>
        `;
    } else if (role === 'error') {
        msgDiv.innerHTML = `<div class="bg-red-500/10 border border-red-500/20 px-4 py-2 rounded-lg text-red-400 text-sm flex items-center gap-2"><i data-lucide="alert-circle" class="w-4 h-4"></i> ${data}</div>`;
    } else {
        // Assistant Message (V2 Intelligent)
        const { answer, context, entry_nodes, thought_process, subgraph, timing } = data;
        
        let thoughtsHtml = '';
        if (thought_process && thought_process.length > 0) {
            thoughtsHtml = `
                <div class="mb-6 p-4 rounded-xl bg-white/5 border border-white/5">
                    <button onclick="this.nextElementSibling.classList.toggle('hidden')" class="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-indigo-400 transition-all uppercase tracking-widest">
                        <i data-lucide="brain" class="w-3.5 h-3.5"></i> Thought Process (${thought_process.length} steps)
                    </button>
                    <div class="mt-4 space-y-4 hidden pt-2">
                        ${thought_process.map(t => `
                            <div class="thought-step">
                                <p class="text-xs font-bold text-indigo-400">Step ${t.step}</p>
                                <p class="text-sm text-slate-400">Searching for: <span class="text-slate-200">${t.nodes.join(', ')}</span></p>
                                <p class="text-[10px] text-slate-500 italic">Retrieved ${t.found_facts} facts</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        const graphId = `graph-${Math.random().toString(36).substr(2, 9)}`;
        
        msgDiv.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-indigo-600 flex-shrink-0 flex items-center justify-center text-white"><i data-lucide="bot" class="w-5 h-5"></i></div>
            <div class="flex-1 space-y-6">
                ${thoughtsHtml}
                
                <div class="prose prose-invert max-w-none text-slate-200 leading-relaxed font-light">
                    ${answer ? formatAnswer(answer) : '(No answer generated)'}
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-white/5">
                    <div class="space-y-3">
                        <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Source Subgraph</p>
                        <div id="${graphId}" class="graph-container"></div>
                    </div>
                    <div class="space-y-3">
                        <p class="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Knowledge Context</p>
                        <div class="p-4 rounded-xl bg-white/5 border border-white/5 text-xs text-slate-400 font-mono h-[300px] overflow-y-auto scrollbar-hide">
                            ${context}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        setTimeout(() => {
            if (subgraph) renderGraph(subgraph, graphId);
            lucide.createIcons();
        }, 100);
    }
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    lucide.createIcons();
}

function formatAnswer(text) {
    return text.replace(/\*\*(.*?)\*\*/g, '<strong class="text-indigo-300">$1</strong>')
               .replace(/\n/g, '<br/>');
}

function toggleLoading(show) {
    sendBtn.innerHTML = show ? `<i data-lucide="loader" class="w-5 h-5 animate-spin"></i>` : `<i data-lucide="arrow-right" class="w-5 h-5"></i>`;
    lucide.createIcons();
}

// ── Visualization Engine (D3.js) ──────────────────────────────────
function renderGraph(subgraph, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const width = container.clientWidth;
    const height = 300;
    
    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("viewBox", [0, 0, width, height]);

    const nodes = subgraph.nodes.map(d => ({...d}));
    const links = subgraph.relationships.map(d => ({
        source: d.source_id,
        target: d.target_id,
        type: d.type
    }));

    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(80))
        .force("charge", d3.forceManyBody().strength(-150))
        .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("class", "link")
        .attr("stroke-width", 1);

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

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function getColor(label) {
        const colors = {
            'Drug': '#6366f1',
            'Disease': '#f43f5e',
            'SideEffect': '#eab308',
            'Manufacturer': '#10b981'
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
