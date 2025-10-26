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
CHUNK_SIZE = 2000 # Characters per chunk for LLM processing
CHUNK_OVERLAP = 300 # Overlap between chunks
MAX_RETRIES = 3  # Max API call retries
TIMEOUT = 30  # API timeout in seconds

# Entity Types
ENTITY_TYPES = [
    # Core Business Entities
    "Company",
    "Person",
    "Product",
    "Subsidiary",
    "Investment",
    "Technology",
    "Location",
    "Event",
    
    # Financial & Strategic Entities
    "FinancialMetric",       # revenue, profit, expenses, EPS
    "FinancialInstrument",   # stocks, bonds, derivatives
    "BusinessUnit",          # divisions like "Apple Services"
    "Market",                # industry or market segments
    "CustomerSegment",       # e.g., "Enterprise Clients", "Individual Consumers"
    "Competitor",            # e.g., "Samsung" in relation to Apple
    "Supplier",              # e.g., "TSMC" for NVIDIA
    
    # Governance & Legal
    "BoardMember",
    "Shareholder",
    "Regulator",
    "Policy",
    "LegalCase",
    
    # Temporal / Event-based
    "Acquisition",
    "Merger",
    "Partnership",
    "IPO",
    "ProductLaunch",
    "Conference"
]

# Relationship Types
RELATIONSHIP_TYPES = [
    # Ownership & Investment
    "OWNS",                 # Company → Subsidiary
    "INVESTS_IN",           # Company → Investment
    "ACQUIRED",             # Company → Company
    "MERGED_WITH",          # Company → Company
    "PARTNERS_WITH",        # Company → Company
    "JOINT_VENTURE_WITH",   # Company → Company

    # Management & Governance
    "CEO_OF",               # Person → Company
    "FOUNDED_BY",           # Company → Person
    "MANAGES",              # Person → BusinessUnit
    "BOARD_MEMBER_OF",      # Person → Company
    "SHAREHOLDER_OF",       # Shareholder → Company
    "REGULATED_BY",         # Company → Regulator
    
    # Supply Chain & Competition
    "SUPPLIES_TO",          # Supplier → Company
    "PROCURES_FROM",        # Company → Supplier
    "COMPETES_WITH",        # Company → Company
    "OPERATES_IN",          # Company → Market
    "LOCATED_IN",           # Company → Location

    # Product & Tech Relationships
    "DEVELOPS",             # Company → Technology/Product
    "USES_TECHNOLOGY",      # Product → Technology
    "LAUNCHES",             # Company → ProductLaunch
    "ANNOUNCED_AT",         # ProductLaunch → Event

    # Financial Relationships
    "HAS_METRIC",           # Company → FinancialMetric
    "ISSUED",               # Company → FinancialInstrument
    "RAISED_CAPITAL_FROM",  # Company → Investor
    "REPORTED_IN",          # FinancialMetric → Report/Year
    "AFFECTED_BY",          # FinancialMetric → Event

    # Legal / Policy
    "SUBJECT_TO",           # Company → Policy
    "INVOLVED_IN",          # Company → LegalCase
]

# Visualization Settings
VIS_PHYSICS_ENABLED = True
VIS_HEIGHT = "750px"
VIS_WIDTH = "100%"
VIS_NOTEBOOK = True

# Logging
LOG_LEVEL = "INFO"