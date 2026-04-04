import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

def load_domain_config(domain: str) -> dict:
    """Load domain-specific JSON config from the config/ directory."""
    config_path = Path(__file__).parent / "config" / f"{domain}_graph.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config for domain '{domain}' not found at {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)