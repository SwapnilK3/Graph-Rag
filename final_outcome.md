# Complete Graph-RAG System Architecture & Implementation Guide

## Executive Summary

This document provides a comprehensive specification for building a **generalizable, traversal-based Graph-RAG memory layer** that retrieves structured knowledge from arbitrary knowledge graphs without using vector embeddings.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Two-Pipeline Design](#two-pipeline-design)
4. [Component Specifications](#component-specifications)
5. [Implementation Details](#implementation-details)
6. [API Design](#api-design)
7. [Configuration Layer](#configuration-layer)
8. [Testing Strategy](#testing-strategy)
9. [Example Use Cases](#example-use-cases)

---

## System Overview

### Core Objective

Build a reusable Graph-RAG memory layer that:
- Works with ANY knowledge graph (schema-agnostic)
- Retrieves information through graph traversal (no embeddings)
- Reflects updates immediately (no re-indexing)
- Provides grounded context to LLMs (LLM only reasons, never retrieves)
- Is plug-and-play (swap graphs without changing code)

### Key Innovation

**Pure traversal-based retrieval** with **configuration-driven abstraction** that separates:
- Graph schema (domain-specific)
- Retrieval logic (domain-agnostic)
- Query understanding (intent-based)

### Core Constraints

- ❌ No vector embeddings
- ❌ No semantic search
- ❌ No manual entry-node selection
- ❌ No agent execution logic
- ✅ Deterministic retrieval
- ✅ Immediate consistency
- ✅ Schema portability

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────┐
│         USER APPLICATION (Future)               │
│    (Chatbot / Agent / Q&A System)               │
└────────────────┬────────────────────────────────┘
                 │
                 │ Uses
                 ↓
┌─────────────────────────────────────────────────┐
│      GRAPH-RAG MEMORY LAYER (Core System)       │
│─────────────────────────────────────────────────│
│                                                  │
│  [1] Entry Node Anchoring                       │
│      Query → Starting Node(s)                   │
│      • Lightweight entity extraction            │
│      • Graph-based validation                   │
│                                                  │
│  [2] Intent Classification                      │
│      • Keyword-based detection                  │
│      • Traversal strategy selection             │
│                                                  │
│  [3] Graph Traversal Engine                     │
│      • Intent-driven navigation                 │
│      • Bounded subgraph extraction              │
│                                                  │
│  [4] Context Extraction                         │
│      • Transform graph → structured text        │
│                                                  │
│  [5] LLM Interface                              │
│      • Provide context to LLM                   │
│      • LLM reasons, never retrieves             │
│                                                  │
└────────────────┬────────────────────────────────┘
                 │
                 │ Reads from
                 ↓
┌─────────────────────────────────────────────────┐
│         CONFIGURATION LAYER                     │
│  • Intent patterns (domain-specific)            │
│  • Traversal rules (per node type)              │
│  • Context templates                            │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│         KNOWLEDGE GRAPH (Neo4j)                 │
│  • Nodes (entities)                             │
│  • Relationships (connections)                  │
│  • Properties (attributes)                      │
└─────────────────────────────────────────────────┘
```

---

## Two-Pipeline Design

### Pipeline 1: Data Ingestion (Offline)

**Purpose:** Build the knowledge graph from documents

```
Documents (PDFs, Text, etc.)
    ↓
Text Extraction
    ↓
Entity-Relationship Extraction (Heavy NLP)
    • Use spaCy, LLMs, or custom models
    • Extract: Entities, Relationships, Properties
    ↓
Store in Neo4j
    • CREATE nodes and relationships
    • Set properties
    ↓
Knowledge Graph Ready
```

**Example:**
```
Document: "Aspirin is a drug that treats headaches but can cause nausea."

Extraction:
- Entity: Aspirin (Drug)
- Entity: Headache (Disease)
- Entity: Nausea (SideEffect)
- Relationship: Aspirin -[TREATS]-> Headache
- Relationship: Aspirin -[CAUSES]-> Nausea

Neo4j Cypher:
CREATE (a:Drug {name: 'Aspirin', type: 'NSAID'})
CREATE (h:Disease {name: 'Headache'})
CREATE (n:SideEffect {name: 'Nausea'})
CREATE (a)-[:TREATS]->(h)
CREATE (a)-[:CAUSES]->(n)
```

**NOT in scope of this system** - assumes graph already exists.

---

### Pipeline 2: Query Processing (Real-time)

**Purpose:** Answer user queries using the graph

```
User Query (via API)
    ↓
[1] Entry Node Anchoring (Lightweight)
    Extract entity keywords → Find matching nodes
    ↓
[2] Intent Classification (Keyword-based)
    Understand what user is asking
    ↓
[3] Graph Traversal (Cypher query)
    Execute intent-specific or exploratory traversal
    ↓
[4] Context Generation
    Convert subgraph to text
    ↓
[5] LLM Response
    Send context to LLM → Get answer
    ↓
Return to User
```

**Example Flow:**
```
Query: "What are the side effects of aspirin?"

[1] Extract: "aspirin" → Find Drug(Aspirin, id=123)
[2] Intent: "side_effects" (keyword: "side effects")
[3] Traverse: Drug(123) -[CAUSES]-> SideEffect
[4] Context: "Aspirin causes Nausea (mild) and Stomach Bleeding (severe)"
[5] LLM: "Aspirin can cause nausea and stomach bleeding..."
```

---

## Component Specifications

### Component 1: Query-Time Entity Extractor

**Purpose:** Find entry nodes from user query (lightweight, fast)

**Input:** User query string  
**Output:** List of graph nodes

**Algorithm:**
```
1. Tokenize query
2. Remove stop words
3. Try exact matches in graph
4. Try partial/fuzzy matches if needed
5. Return matching nodes
```

**Code Structure:**
```python
class QueryTimeEntityExtractor:
    def __init__(self, connector):
        self.connector = connector
        self.stop_words = {...}
    
    def extract_entry_nodes(self, query: str) -> list:
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Find matching nodes
        entry_nodes = []
        for keyword in keywords:
            nodes = self._find_in_graph(keyword)
            entry_nodes.extend(nodes)
        
        return entry_nodes
    
    def _extract_keywords(self, query: str) -> list:
        # Tokenize, remove stop words
        # Handle multi-word entities
        pass
    
    def _find_in_graph(self, keyword: str) -> list:
        # Cypher: MATCH (n) WHERE toLower(n.name) = $keyword
        # Try exact, then partial match
        pass
```

**Key Features:**
- No heavy NLP libraries needed
- Fast execution (< 100ms)
- Validates entities exist in graph
- Handles multi-word entities

---

### Component 2: Intent Classifier

**Purpose:** Understand query intent to guide traversal

**Input:** User query string  
**Output:** Intent type (side_effects, treatment, interaction, general)

**Algorithm:**
```
1. Define intent keyword patterns
2. Check if query contains keywords
3. Return matched intent or "general"
```

**Code Structure:**
```python
class QueryIntentClassifier:
    INTENT_KEYWORDS = {
        "side_effects": ["side effect", "adverse", "risk", "cause"],
        "treatment": ["treat", "cure", "help", "medicine for"],
        "interaction": ["interact", "combine", "take with"],
        "general": ["about", "information", "tell me"]
    }
    
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        
        return "general"
```

**Configurable:** Load intent patterns from JSON config file

---

### Component 3: Smart Traversal Engine

**Purpose:** Execute graph queries based on intent

**Input:** Entry nodes + Intent  
**Output:** Subgraph (nodes + relationships)

**Strategies:**

#### Strategy 1: Targeted Relationship Query
```python
def _traverse_side_effects(self, entry_nodes):
    entry_id = entry_nodes[0]['id']
    
    query = """
    MATCH (drug)-[r:CAUSES]->(se:SideEffect)
    WHERE id(drug) = $entry_id
    RETURN drug, se, r
    """
    
    return self.execute_and_format(query, {"entry_id": entry_id})
```

#### Strategy 2: Reverse Traversal
```python
def _traverse_treatments(self, entry_nodes):
    # If entry is Disease, find Drugs that treat it
    query = """
    MATCH (drug:Drug)-[r:TREATS]->(disease)
    WHERE id(disease) = $entry_id
    RETURN drug, disease, r
    """
```

#### Strategy 3: Multi-Entity Relationship
```python
def _traverse_interactions(self, entry_nodes):
    # Check relationship between two entities
    query = """
    MATCH (d1)-[r:INTERACTS_WITH]-(d2)
    WHERE id(d1) = $id1 AND id(d2) = $id2
    RETURN d1, d2, r
    """
```

#### Strategy 4: General Exploration
```python
def _traverse_general(self, entry_nodes):
    # Multi-hop traversal (bounded)
    query = """
    MATCH path = (start)-[*1..2]-(connected)
    WHERE id(start) = $entry_id
    RETURN nodes(path), relationships(path)
    LIMIT 50
    """
```

---

### Component 4: Context Generator

**Purpose:** Convert graph data to LLM-readable text

**Input:** Subgraph (nodes + relationships)  
**Output:** Structured text context

**Template-Based Approach:**
```python
class ContextGenerator:
    def generate(self, subgraph: dict) -> str:
        context_parts = []
        
        # Add nodes
        context_parts.append("ENTITIES:")
        for node in subgraph['nodes']:
            context_parts.append(
                f"- {node['properties']['name']} ({node['label']})"
            )
        
        # Add relationships
        context_parts.append("\nRELATIONSHIPS:")
        for rel in subgraph['relationships']:
            source = self._get_node_name(rel['source'], subgraph)
            target = self._get_node_name(rel['target'], subgraph)
            context_parts.append(
                f"- {source} {rel['type']} {target}"
            )
        
        return "\n".join(context_parts)
```

**Example Output:**
```
ENTITIES:
- Aspirin (Drug)
- Nausea (SideEffect)
- Stomach Bleeding (SideEffect)

RELATIONSHIPS:
- Aspirin CAUSES Nausea
- Aspirin CAUSES Stomach Bleeding
```

---

### Component 5: LLM Interface

**Purpose:** Send context to LLM and get grounded response

**Input:** Query + Context  
**Output:** LLM answer

**Code Structure:**
```python
class LLMInterface:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def answer(self, query: str, context: str) -> str:
        prompt = f"""
Based ONLY on the following knowledge graph information, answer the question.
Do not use external knowledge.

KNOWLEDGE GRAPH CONTEXT:
{context}

QUESTION: {query}

ANSWER:
        """
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
```

**Critical Constraint:** LLM must ONLY use provided context, never external knowledge

---

## Implementation Details

### Technology Stack

**Required:**
- Python 3.8+
- Neo4j 5.x (graph database)
- neo4j Python driver
- Anthropic API (or OpenAI)

**Optional:**
- Flask/FastAPI (API framework)
- python-Levenshtein (fuzzy matching)
- python-dotenv (config management)

### Project Structure

```
graph-rag-project/
├── src/
│   ├── __init__.py
│   ├── connector.py      # Neo4j connection
│   ├── entity_extractor.py     # Query-time entity extraction
│   ├── intent_classifier.py    # Intent detection
│   ├── traversal_engine.py     # Graph traversal logic
│   ├── context_generator.py    # Context creation
│   └── llm_interface.py        # LLM integration
├── config/
│   ├── medical_graph.json      # Medical domain config
│   ├── movie_graph.json        # Movie domain config
│   └── config_template.json    # Template for new domains
├── api/
│   └── app.py                  # API endpoints
├── tests/
│   ├── test_extractor.py
│   ├── test_traversal.py
│   └── test_integration.py
├── .env                        # Environment variables
├── requirements.txt
└── README.md
```

### Environment Configuration

**.env file:**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
ANTHROPIC_API_KEY=your_api_key
```

### Dependencies

**requirements.txt:**
```
neo4j==5.14.0
python-dotenv==1.0.0
anthropic==0.18.1
flask==3.0.0
python-Levenshtein==0.21.1
```

---

## API Design

### REST API Endpoint

**Endpoint:** `POST /api/query`

**Request:**
```json
{
  "query": "What are the side effects of aspirin?",
  "options": {
    "max_hops": 2,
    "include_context": true
  }
}
```

**Response:**
```json
{
  "query": "What are the side effects of aspirin?",
  "answer": "Aspirin can cause nausea (mild) and stomach bleeding (severe).",
  "metadata": {
    "entry_nodes": ["Aspirin"],
    "intent": "side_effects",
    "nodes_retrieved": 3,
    "execution_time_ms": 245
  },
  "context": "Aspirin is a Drug...",
  "debug": {
    "cypher_query": "MATCH (drug)-[:CAUSES]->(se:SideEffect)...",
    "raw_results": [...]
  }
}
```

### API Implementation

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize components
connector = GraphConnector()
extractor = QueryTimeEntityExtractor(connector)
intent_classifier = QueryIntentClassifier()
traversal = SmartTraversalEngine(connector)
context_gen = ContextGenerator()
llm = LLMInterface(api_key=os.getenv('ANTHROPIC_API_KEY'))

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')
    
    # Extract entry nodes
    entry_nodes = extractor.extract_entry_nodes(user_query)
    
    if not entry_nodes:
        return jsonify({"error": "No entities found"}), 404
    
    # Classify intent
    intent = intent_classifier.classify(user_query)
    
    # Traverse graph
    subgraph = traversal.traverse_by_intent(entry_nodes, intent)
    
    # Generate context
    context = context_gen.generate(subgraph)
    
    # Get LLM answer
    answer = llm.answer(user_query, context)
    
    return jsonify({
        "query": user_query,
        "answer": answer,
        "metadata": {
            "entry_nodes": [n['name'] for n in entry_nodes],
            "intent": intent,
            "nodes_retrieved": len(subgraph['nodes'])
        },
        "context": context
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## Configuration Layer

### Purpose

Make the system work with different graph schemas without code changes

### Configuration File Structure

**config/medical_graph.json:**
```json
{
  "domain": "medical",
  "description": "Medical knowledge graph configuration",
  
  "node_types": {
    "Drug": {
      "search_properties": ["name", "generic_name", "brand_name"],
      "display_properties": ["name", "type", "description"]
    },
    "Disease": {
      "search_properties": ["name", "common_name"],
      "display_properties": ["name", "severity"]
    },
    "SideEffect": {
      "search_properties": ["name"],
      "display_properties": ["name", "severity"]
    }
  },
  
  "intent_patterns": {
    "side_effects": {
      "keywords": ["side effect", "adverse", "risk", "cause"],
      "relationship": "CAUSES",
      "direction": "OUTGOING",
      "source_type": "Drug",
      "target_type": "SideEffect"
    },
    "treatment": {
      "keywords": ["treat", "cure", "help", "medicine for"],
      "relationship": "TREATS",
      "direction": "OUTGOING",
      "source_type": "Drug",
      "target_type": "Disease"
    },
    "treated_by": {
      "keywords": ["what treats", "treatment for"],
      "relationship": "TREATS",
      "direction": "INCOMING",
      "source_type": "Disease",
      "target_type": "Drug"
    }
  },
  
  "traversal_rules": {
    "Drug": {
      "max_hops": 2,
      "relationship_types": ["TREATS", "CAUSES", "INTERACTS_WITH"],
      "node_limit": 50
    },
    "Disease": {
      "max_hops": 1,
      "relationship_types": ["TREATS", "SYMPTOM_OF"],
      "node_limit": 30
    }
  },
  
  "context_templates": {
    "Drug": "{name} is a {type} drug",
    "Disease": "{name} (severity: {severity})",
    "TREATS": "{source} treats {target}",
    "CAUSES": "{source} may cause {target}"
  }
}
```

### Loading Configuration

```python
import json

class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
    
    def get_searchable_properties(self, node_type):
        return self.config['node_types'][node_type]['search_properties']
    
    def get_intent_pattern(self, intent_name):
        return self.config['intent_patterns'][intent_name]
    
    def get_traversal_rules(self, node_type):
        return self.config['traversal_rules'][node_type]
```

---

## Testing Strategy

### Unit Tests

**Test Entity Extraction:**
```python
def test_entity_extraction():
    query = "What are the side effects of aspirin?"
    extractor = QueryTimeEntityExtractor(connector)
    nodes = extractor.extract_entry_nodes(query)
    
    assert len(nodes) == 1
    assert nodes[0]['name'] == 'Aspirin'
    assert nodes[0]['label'] == 'Drug'
```

**Test Intent Classification:**
```python
def test_intent_classification():
    classifier = QueryIntentClassifier()
    
    assert classifier.classify("side effects of aspirin") == "side_effects"
    assert classifier.classify("what treats headaches") == "treated_by"
    assert classifier.classify("tell me about aspirin") == "general"
```

### Integration Tests

**End-to-End Query Test:**
```python
def test_full_query_flow():
    query = "What are the side effects of aspirin?"
    
    # Execute full pipeline
    response = api_client.post('/api/query', json={"query": query})
    
    assert response.status_code == 200
    assert "nausea" in response.json['answer'].lower()
    assert response.json['metadata']['intent'] == "side_effects"
```

### Test Queries

**Medical Domain:**
- "What are the side effects of aspirin?"
- "What treats headaches?"
- "Can I take aspirin with warfarin?"
- "Tell me about ibuprofen"

**Expected Behaviors:**
- Query 1: Returns only SideEffect nodes
- Query 2: Returns Drug nodes that treat Headache
- Query 3: Returns INTERACTS_WITH relationship
- Query 4: Returns all related information

---

## Example Use Cases

### Use Case 1: Medical Knowledge Assistant

**Scenario:** Doctor wants to check drug information

**Query:** "What are the contraindications for aspirin?"

**System Flow:**
1. Extract: "aspirin" → Drug(Aspirin)
2. Intent: "side_effects" (close enough)
3. Traverse: Get CAUSES relationships
4. Context: "Aspirin causes bleeding, nausea..."
5. Answer: "Aspirin should not be used if patient has bleeding disorders..."

---

### Use Case 2: Treatment Recommendation

**Scenario:** Patient asks about treatment options

**Query:** "What can I take for arthritis?"

**System Flow:**
1. Extract: "arthritis" → Disease(Arthritis)
2. Intent: "treated_by"
3. Traverse: Disease ←[TREATS]- Drug
4. Context: "Ibuprofen treats Arthritis, Naproxen treats Arthritis..."
5. Answer: "For arthritis, options include Ibuprofen and Naproxen..."

---

### Use Case 3: Drug Interaction Check

**Scenario:** Pharmacist verifies drug combination

**Query:** "Is it safe to take aspirin with warfarin?"

**System Flow:**
1. Extract: ["aspirin", "warfarin"] → [Drug(Aspirin), Drug(Warfarin)]
2. Intent: "interaction"
3. Traverse: Drug ←[INTERACTS_WITH]→ Drug
4. Context: "Aspirin INTERACTS_WITH Warfarin (severe - increased bleeding risk)"
5. Answer: "No, aspirin and warfarin have a severe interaction..."

---

## Generalization to Other Domains

### Movie Database Example

**Graph Schema:**
- Nodes: Movie, Actor, Director, Genre
- Relationships: ACTED_IN, DIRECTED, BELONGS_TO

**Config: config/movie_graph.json:**
```json
{
  "domain": "movies",
  "intent_patterns": {
    "actor_movies": {
      "keywords": ["movies with", "acted in", "starred in"],
      "relationship": "ACTED_IN",
      "direction": "OUTGOING",
      "source_type": "Actor",
      "target_type": "Movie"
    },
    "movie_cast": {
      "keywords": ["who acted", "cast of", "actors in"],
      "relationship": "ACTED_IN",
      "direction": "INCOMING",
      "source_type": "Movie",
      "target_type": "Actor"
    }
  }
}
```

**Example Query:**
```
Query: "What movies has Tom Hanks acted in?"

Flow:
1. Extract: "Tom Hanks" → Actor(Tom Hanks)
2. Intent: "actor_movies"
3. Traverse: Actor -[ACTED_IN]-> Movie
4. Context: "Tom Hanks acted in Forrest Gump, Cast Away..."
5. Answer: "Tom Hanks has starred in Forrest Gump, Cast Away..."
```

**Key Point:** SAME CODE, different config!

---

## Performance Considerations

### Query Optimization

**Indexed Properties:**
```cypher
CREATE INDEX node_name_index FOR (n:Drug) ON (n.name)
CREATE INDEX node_name_index FOR (n:Disease) ON (n.name)
```

**Bounded Traversal:**
- Limit max hops (1-2 for most queries)
- Limit result count (LIMIT 50)
- Use specific relationship types

### Caching Strategy

**Cache frequent queries:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_traversal(entry_node_id, intent):
    return traversal_engine.traverse(entry_node_id, intent)
```

### Scalability

**For large graphs (1M+ nodes):**
- Use targeted queries (not general exploration)
- Implement pagination
- Consider graph partitioning
- Add query timeouts

---

## Future Enhancements

### Phase 2 Improvements

1. **Fuzzy Matching:** Handle typos in entity names
2. **Multi-word Entity Detection:** Better extraction of "stomach bleeding"
3. **Synonym Handling:** Map "acetylsalicylic acid" → "Aspirin"
4. **Confidence Scoring:** Rank multiple entity matches

### Phase 3 Advanced Features

1. **Hybrid Retrieval:** Combine traversal + vector search
2. **Multi-hop Reasoning:** Complex queries requiring 3+ hops
3. **Temporal Queries:** "What was known about aspirin in 2020?"
4. **Aggregation:** "How many drugs treat headaches?"

### Phase 4 Production Ready

1. **Authentication & Authorization**
2. **Rate Limiting**
3. **Monitoring & Logging**
4. **A/B Testing Framework**
5. **Performance Benchmarking**

---

## Success Metrics

### Functional Metrics

- **Entity Recognition Accuracy:** >90% of queries find correct entry nodes
- **Intent Classification Accuracy:** >85% correct intent detection
- **Answer Relevance:** >90% of answers use only retrieved context
- **Query Success Rate:** >95% of valid queries return results

### Performance Metrics

- **Query Latency:** <500ms end-to-end (p95)
- **Graph Query Time:** <100ms (p95)
- **LLM Response Time:** <2s (p95)
- **Throughput:** >100 queries/second

### Generalization Metrics

- **Config-Only Domain Switch:** Add new domain in <1 hour
- **Zero Code Changes:** Swap domains without touching code
- **Schema Coverage:** Works with 3+ different graph schemas

---

## Conclusion

This Graph-RAG system provides a **generalizable, deterministic, and immediately consistent** approach to knowledge retrieval from structured graphs. By separating domain knowledge (configuration) from retrieval logic (code), it achieves true portability across different knowledge domains while maintaining high performance and accuracy.

The key innovation is the **intent-driven traversal** approach that uses the graph's inherent structure rather than embedding-based similarity, ensuring reproducibility and transparency in the retrieval process.

---

## Appendix: Quick Start Commands

### Setup Neo4j
```bash
# Download Neo4j Desktop from https://neo4j.com/download/
# Create database, set password
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Create Sample Graph
```cypher
CREATE (a:Drug {name: 'Aspirin', type: 'NSAID'})
CREATE (h:Disease {name: 'Headache', severity: 'Mild'})
CREATE (n:SideEffect {name: 'Nausea', severity: 'Mild'})
CREATE (a)-[:TREATS]->(h)
CREATE (a)-[:CAUSES]->(n)
```

### Run API
```bash
python api/app.py
```

### Test Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the side effects of aspirin?"}'
```

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Graph-RAG System Specification  
**License:** MIT

---

# End of Document

Save this document and share with your team or other AI models for implementation guidance.