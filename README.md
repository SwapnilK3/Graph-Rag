# Graph-RAG — Knowledge Graph Retrieval-Augmented Generation

A **config-driven**, multi-hop RAG system that answers natural-language questions
by retrieving structured facts from a **Neo4j knowledge graph** and handing those
facts to a **Gemini LLM** — with zero hallucination risk, because the LLM is only
allowed to use what the graph returns.

---

## Table of Contents

1. [What Problem Does This Solve?](#what-problem-does-this-solve)
2. [How It Works — Big Picture](#how-it-works--big-picture)
3. [Architecture Overview](#architecture-overview)
4. [Component Deep Dive](#component-deep-dive)
   - [Component 1 — Entity Extractor](#component-1--entity-extractor)
   - [Component 2 — Intent Classifier](#component-2--intent-classifier)
   - [Component 3 — Traversal Engine (5 Strategies)](#component-3--traversal-engine)
   - [Component 4 — Context Generator](#component-4--context-generator)
   - [Component 5 — LLM Interface](#component-5--llm-interface)
5. [Domain Config File](#domain-config-file)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [Running the Tests](#running-the-tests)
9. [Adding a New Domain](#adding-a-new-domain)

---

## What Problem Does This Solve?

A plain LLM answers from its training data — it can be stale, wrong, or
unverifiable. A plain vector-search RAG is good for documents but cannot follow
**relationships** ("What disease does this drug's side effect increase the risk
of?").

Graph-RAG combines the best of both:

| Concern | Solution |
|---------|----------|
| Up-to-date, structured facts | Stored in a Neo4j knowledge graph |
| Multi-hop reasoning | Graph traversal at query time |
| Natural-language answers | Gemini LLM reads retrieved graph context |
| Grounded answers only | LLM is forbidden from using external knowledge |

---

## How It Works — Big Picture

```
User query
    │
    ▼
[1] Entity Extractor   ─── finds named nodes in the graph
    │                      (e.g. "aspirin" → Drug node)
    ▼
[2] Intent Classifier  ─── decides WHY the user is asking
    │                      (e.g. "side effects" vs "risk chain")
    ▼
[3] Traversal Engine   ─── walks the graph to collect relevant facts
    │                      (1-hop / multi-hop / shortest-path / …)
    ▼
[4] Context Generator  ─── converts the subgraph to readable sentences
    │                      (e.g. "Aspirin may cause Stomach Bleeding")
    ▼
[5] LLM Interface      ─── Gemini reads the context and answers in plain English
    │
    ▼
Answer
```

A single query like *"What diseases does aspirin lead to through its side
effects?"* follows a 2-hop path automatically:

```
Aspirin ──CAUSES──▶ Stomach Bleeding ──INCREASES_RISK_OF──▶ Peptic Ulcer
```

Gemini then says: *"Aspirin leads to Peptic Ulcer through its side effect,
Stomach Bleeding."*

---

## Architecture Overview

```
e:\Project\Graph-Rag\
│
├── config/
│   └── medical_graph.json   ← single file that controls ALL behaviour
│
├── connector.py             ← Neo4j connection wrapper
├── entity_extractor.py      ← Component 1
├── intent_classifier.py     ← Component 2
├── traversal_engine.py      ← Component 3
├── context_generator.py     ← Component 4
├── llm_interface.py         ← Component 5
│
└── tests/
    ├── test_components.py        ← 35 offline unit tests
    ├── test_hopping.py           ← 57 offline multi-hop strategy tests
    ├── test_pipeline.py          ← 5 live tests (targeted strategies)
    └── test_complex_pipeline.py  ← 8 live tests (all 5 strategies + LLM)
```

The entire system is **config-driven**: swap `medical_graph.json` for
`legal_graph.json` and the same five Python files work in a completely
different domain without code changes.

---

## Component Deep Dive

### Component 1 — Entity Extractor

**File:** `entity_extractor.py`  
**Class:** `QueryTimeEntityExtractor`

**Job:** Find real graph nodes mentioned in the user's query.

**How:**
1. Splits the query into n-grams: trigrams first, then bigrams, then single words.
   Longer phrases are tried first so "stomach bleeding" beats "stomach" alone.
2. Removes stop-words ("the", "what", "does", etc.).
3. For each candidate, queries Neo4j with:
   - **Exact match** (case-insensitive `CONTAINS`)
   - **Partial match** (substring)
   - **Fuzzy match** (Levenshtein distance — no typo goes unmatched)
4. Returns a list of matched node dicts: `{id, label, name, properties}`.

These nodes are the **entry points** for graph traversal.

---

### Component 2 — Intent Classifier

**File:** `intent_classifier.py`  
**Class:** `IntentClassifier`

**Job:** Decide what the user wants to know.

**How:** Scans the query for keywords defined in the config file.
The first matching intent wins. No ML — fast and fully deterministic.

| Intent | Example trigger words | What it does |
|--------|-----------------------|--------------|
| `side_effects` | "side effect", "cause" | 1-hop Drug → SideEffect |
| `treatment` | "treat", "cure" | 1-hop Drug → Disease |
| `treated_by` | "what treats", "what drug" | 1-hop Disease → Drug (reverse) |
| `interaction` | "interact", "take with" | 1-hop Drug ↔ Drug |
| `symptoms` | "symptom", "signs" | 1-hop Disease → Symptom |
| `indirect_symptoms` | "indirectly", "diseases it treats" | 2-hop Drug→Disease→Symptom |
| `drug_risk_chain` | "lead to", "risk chain" | 2-hop Drug→SideEffect→Disease |
| `neighborhood` | "overview", "all about" | 1–2 hop free exploration |
| `connection` | "connected", "path between" | shortest path between nodes |
| `shared_effects` | "share", "both cause", "in common" | shared neighbors |

> **Order matters:** More-specific multi-hop intents are listed before
> overlapping 1-hop intents in the config to avoid false matches.

---

### Component 3 — Traversal Engine

**File:** `traversal_engine.py`  
**Class:** `SmartTraversalEngine`

**Job:** Run a Cypher query against Neo4j and return a standardised subgraph.

**Five strategies** — each produces the same output shape so the rest of the
pipeline is unaffected:

#### Strategy 1: `targeted` (1-hop)
The simplest strategy. Follows one named relationship in one direction.

```
Config:   relationship=CAUSES, anchor=source
Entry:    Aspirin (Drug)
Cypher:   MATCH (src:Drug {name:"Aspirin"})-[r:CAUSES]->(tgt) RETURN …
Result:   Aspirin → Nausea, Aspirin → Stomach Bleeding
```

#### Strategy 2: `chained` (fixed multi-hop)
Follows a predetermined sequence of relationships — like a recipe.

```
Config:   hops = [CAUSES → SideEffect, INCREASES_RISK_OF → Disease]
Entry:    Aspirin
Cypher:   MATCH (n0)-[:CAUSES]->(n1)-[:INCREASES_RISK_OF]->(n2)
          WHERE n0.name = "Aspirin"
Result:   Aspirin → Stomach Bleeding → Peptic Ulcer  (depth = 2)
```

#### Strategy 3: `variable_hop` (free N-hop)
Explores freely up to N hops in any direction — good for "give me everything
about X".

```
Config:   min_hops=1, max_hops=2
Entry:    Metformin
Cypher:   MATCH path = (src)-[*1..2]-(tgt) WHERE src.name = "Metformin"
Result:   Metformin + all 1-hop and 2-hop neighbours
```

#### Strategy 4: `shortest_path`
Finds the shortest connection between two (or more) entry nodes.

```
Entry:    Aspirin, Peptic Ulcer
Cypher:   MATCH path = shortestPath((a)-[*..6]-(b))
          WHERE a.name="Aspirin" AND b.name="Peptic Ulcer"
Result:   Aspirin → Stomach Bleeding → Peptic Ulcer  (length 2)
```

If only one entry node is found, falls back to `variable_hop` automatically.

#### Strategy 5: `shared_neighbor`
Finds nodes reachable from **all** entry nodes (intersection).

```
Entry:    Aspirin, Ibuprofen
Config:   min_connections = 2
Cypher:   MATCH (entry)--(shared) WHERE entry.name IN ["Aspirin","Ibuprofen"]
          WITH shared, count(DISTINCT entry) AS connected_entries
          WHERE connected_entries >= 2
Result:   Nausea  (caused by both)
```

**Output format — every strategy returns:**
```python
{
    "nodes":         [{"id", "label", "name", "properties"}, ...],
    "relationships": [{"source_id", "target_id", "type", "properties"}, ...],
    "strategy":      "chained",   # which strategy ran
    "hop_depth":     2,           # max hops traversed
}
```

---

### Component 4 — Context Generator

**File:** `context_generator.py`  
**Class:** `ContextGenerator`

**Job:** Turn the raw subgraph dict into readable English that an LLM can use.

**Input:** subgraph dict from Traversal Engine  
**Output:**
```
[Traversal: chained, depth=2, 3 nodes, 2 edges]

ENTITIES:
  - [Drug] Aspirin (type: NSAID)
  - [SideEffect] Stomach Bleeding (severity: Severe)
  - [Disease] Peptic Ulcer (severity: Moderate)

RELATIONSHIPS:
  - Aspirin may cause Stomach Bleeding
  - Stomach Bleeding increases risk of Peptic Ulcer
```

Relationship sentences use **templates from the config file** so you always
get domain-appropriate language rather than raw Cypher arrow notation.

---

### Component 5 — LLM Interface

**File:** `llm_interface.py`  
**Class:** `LLMInterface`

**Job:** Send context + query to Gemini and return the answer.

**Key constraint — the system prompt is:**
> *"Answer the question using ONLY the information provided in the KNOWLEDGE
> GRAPH CONTEXT. Do not use any external knowledge."*

This makes every answer **fully traceable** — if a fact is not in the graph,
Gemini says so rather than making something up.

**Model:** `gemini-2.5-flash` (fast, cost-efficient, handles structured text well)

---

## Domain Config File

`config/medical_graph.json` is the **single control file** for the entire system.

```
{
  "domain": "medical",

  "intent_patterns": {          ← what keywords trigger which traversal
    "side_effects": {
      "strategy": "targeted",
      "keywords": ["side effect", "cause", ...],
      "relationship": "CAUSES",
      "entry_anchor": "source"
    },
    "drug_risk_chain": {
      "strategy": "chained",
      "keywords": ["lead to", "risk chain", ...],
      "hops": [
        {"relationship": "CAUSES",            "target_label": "SideEffect"},
        {"relationship": "INCREASES_RISK_OF", "target_label": "Disease"}
      ]
    },
    ...
  },

  "relationship_templates": {   ← how to describe each edge in plain English
    "TREATS":            "{source} treats {target}",
    "CAUSES":            "{source} may cause {target}",
    "INCREASES_RISK_OF": "{source} increases risk of {target}",
    ...
  }
}
```

**To support a new domain** (e.g. legal, financial), create a new JSON file
with different node labels, relationships, and intent keywords.
The five Python components need **zero changes**.

---

## Project Structure

```
Graph-Rag/
│
├── .env                          # Credentials (never commit this)
│   ├── NEO4J_URI                 # neo4j+ssc://… (Aura TLS)
│   ├── NEO4J_USERNAME
│   ├── NEO4J_PASSWORD
│   ├── NEO4J_DATABASE
│   └── GEMINI_API_KEY
│
├── config/
│   └── medical_graph.json        # All domain logic lives here
│
├── config.py                     # Loads .env into Python variables
├── connector.py                  # Neo4j driver wrapper
├── graph_connector.py            # Backward-compat re-export of connector.py
│
├── entity_extractor.py           # Component 1 — finds entry nodes
├── intent_classifier.py          # Component 2 — classifies query intent
├── traversal_engine.py           # Component 3 — 5-strategy graph walker
├── context_generator.py          # Component 4 — subgraph → text
├── llm_interface.py              # Component 5 — Gemini Q&A
│
├── test_components.py            # 35 offline unit tests (no DB / LLM needed)
├── test_hopping.py               # 57 offline multi-hop strategy tests
├── test_pipeline.py              # 5 live tests (targeted intents)
└── test_complex_pipeline.py      # 8 live tests (all 5 strategies + Gemini)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A [Neo4j Aura](https://neo4j.com/cloud/aura/) free instance
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier: 5 RPM)

### Steps

```bash
# 1. Clone / open the project
cd "e:\Project\Graph-Rag"

# 2. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env with your credentials
# NEO4J_URI=neo4j+ssc://<your-instance>.databases.neo4j.io
# NEO4J_USERNAME=<your-username>
# NEO4J_PASSWORD=<your-password>
# NEO4J_DATABASE=<your-database>
# GEMINI_API_KEY=<your-api-key>
```

### requirements.txt

```
neo4j==6.1.0
python-dotenv==1.2.1
google-genai>=1.0.0
python-Levenshtein==0.27.3
pytest==9.0.2
```

> **Why `neo4j+ssc://`?**  
> Neo4j Aura uses TLS with a self-signed certificate chain.  
> `neo4j+ssc://` enables TLS but skips certificate verification — the only
> scheme that works reliably with Aura on the free tier.

---

## Running the Tests

```bash
# Offline tests — no DB or API key required (fast, ~3 s)
.\.venv\Scripts\python.exe -m pytest test_components.py test_hopping.py -v

# Live targeted tests — needs Neo4j Aura connection
.\.venv\Scripts\python.exe -m pytest test_pipeline.py -v

# Live multi-hop tests — needs Neo4j + Gemini API key
# Seeds a rich 32-node medical graph and runs all 5 strategies
.\.venv\Scripts\python.exe .\test_complex_pipeline.py
```

### Test coverage summary

| Test file | Count | Requires |
|-----------|-------|---------|
| `test_components.py` | 35 | nothing (all mocked) |
| `test_hopping.py` | 57 | nothing (all mocked) |
| `test_pipeline.py` | 5 | Neo4j Aura |
| `test_complex_pipeline.py` | 8 + 1 bonus | Neo4j Aura + Gemini |

---

## Adding a New Domain

1. Create `config/my_domain.json` with your node labels, relationships, and intent patterns.
2. Seed your Neo4j instance with nodes and relationships matching those labels.
3. Point the pipeline at the new config:

```python
CONFIG = "config/my_domain.json"

extractor  = QueryTimeEntityExtractor(connector, search_properties=["name"])
classifier = IntentClassifier(CONFIG)
engine     = SmartTraversalEngine(connector, CONFIG)
gen        = ContextGenerator(CONFIG)
llm        = LLMInterface()

entry_nodes = extractor.extract_entry_nodes(query)
intent      = classifier.classify(query)
subgraph    = engine.traverse(entry_nodes, intent)
context     = gen.generate(subgraph)
answer      = llm.answer(query, context)
print(answer)
```

No other code changes are needed.
