### Detailed Algorithm for Multi-Factor Scoring:
```python
def identify_searchable_properties(label, properties):
    for prop in properties:
        score = 0
        
        # Check 1: Name pattern
        if prop.lower() in ["name", "title", "label"]:
            score += 0.4
            
        # Check 2: Cardinality
        unique_count = query("MATCH (n:{label}) RETURN count(DISTINCT n.{prop})")
        total_count = query("MATCH (n:{label}) RETURN count(n)")
        if unique_count / total_count > 0.5:
            score += 0.3
            
        # Check 3: Type
        if is_string_property(label, prop):
            score += 0.3
            
        if score > 0.5:
            searchable.append(prop)
```
```

---

## ✅ **WHAT THE AI WILL DO**

When you paste this prompt, the AI will:

1. ✅ Create the project structure
2. ✅ Implement Component 1 with full code
3. ✅ Add logging and error handling
4. ✅ Write docstrings and type hints
5. ✅ Ask clarifying questions if needed
6. ✅ Wait for your approval before moving to Component 2

---

## 🚀 **FOLLOW-UP PROMPTS**

### **After Component 1 is Done:**
```
Great! Now let's implement Component 2: UniversalEntityExtractor. 
Make sure it uses the schema from Component 1 and searches across ALL node types.
```

### **After All Components:**
```
Now create a main.py file that:
1. Loads environment variables
2. Initializes the pipeline
3. Runs my 5 test queries
4. Prints results in a nice format
```

### **For Testing:**
```
Create a comprehensive test suite using pytest. Include:
- Mock tests (don't need Neo4j)
- Integration tests (need Neo4j running)
- End-to-end test with all 5 components