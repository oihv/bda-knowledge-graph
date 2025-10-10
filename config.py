"""
Configuration file for the Financial Knowledge Graph System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they don't exist
for directory in [DATA_DIR, SAMPLE_REPORTS_DIR, PROCESSED_DIR, OUTPUT_DIR, VISUALIZATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# LLM Configuration (OpenRouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models available on OpenRouter
# Options: "meta-llama/llama-3.1-8b-instruct:free", "google/gemma-2-9b-it:free", etc.
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# Processing Configuration
CHUNK_SIZE = 2000  # Characters per chunk for LLM processing
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_RETRIES = 3  # Max API call retries
TIMEOUT = 30  # API timeout in seconds

# Entity Types
ENTITY_TYPES = [
    "Company",
    "Person",
    "Product",
    "Investment",
    "Subsidiary",
    "Technology",
    "Location",
    "Event",
    "FinancialMetric"
]

# Relationship Types
RELATIONSHIP_TYPES = [
    "OWNS",
    "INVESTS_IN",
    "MANAGES",
    "CEO_OF",
    "FOUNDED_BY",
    "ACQUIRED",
    "PARTNERS_WITH",
    "COMPETES_WITH",
    "SUPPLIES_TO",
    "LOCATED_IN",
    "DEVELOPS",
    "HAS_METRIC"
]

# Visualization Settings
VIS_PHYSICS_ENABLED = True
VIS_HEIGHT = "750px"
VIS_WIDTH = "100%"
VIS_NOTEBOOK = True

# Logging
LOG_LEVEL = "INFO"