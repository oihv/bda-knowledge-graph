#!/usr/bin/env python3
"""
Smart Financial Knowledge Graph Pipeline
Automatically detects if documents are already processed and skips extraction
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from colorama import init, Fore, Style

# Import the original main functions
from main import (
    step_1_preprocessing, step_2_extraction, step_3_graph_construction, 
    step_4_visualization, step_5_query_interface
)
from config import OUTPUT_DIR, SAMPLE_REPORTS_DIR
from src.graph_builder.neo4j_client import Neo4jClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

init(autoreset=True)  # Initialize colorama

def check_document_processed(document_name: str) -> bool:
    """
    Check if a document has already been processed by looking at Neo4j
    
    Args:
        document_name: Name of the document to check
        
    Returns:
        True if document appears to be processed, False otherwise
    """
    try:
        client = Neo4jClient()
        client.connect()
        
        # Check for document nodes or significant data that suggests processing
        stats = client.get_graph_statistics()
        total_entities = sum(stats.get('node_types', {}).values())
        total_relationships = sum(stats.get('relationship_types', {}).values())
        
        # If we have substantial data, assume documents are processed
        # This is a simple heuristic - you could make it more sophisticated
        if total_entities > 50 and total_relationships > 50:
            print(f"{Fore.GREEN}✓ Found existing data in Neo4j:")
            print(f"  - {total_entities} entities across {len(stats.get('node_types', {}))} types")
            print(f"  - {total_relationships} relationships across {len(stats.get('relationship_types', {}))} types")
            
            # Check for specific document indicators
            doc_query = "MATCH (d:Document) RETURN d.name as name LIMIT 5"
            docs = client.execute_cypher(doc_query)
            if docs:
                print(f"  - Documents: {[d['name'] for d in docs]}")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"{Fore.YELLOW}⚠ Could not check Neo4j for existing data: {e}")
        return False

def main():
    """Main pipeline with smart document processing detection"""
    
    logger.info("Starting smart financial knowledge graph pipeline")
    
    parser = argparse.ArgumentParser(description="Smart Financial Knowledge Graph Pipeline")
    parser.add_argument("--input", "-i", help="Input directory or file path")
    parser.add_argument("--use-sample", action="store_true", 
                       help="Use sample documents from data/sample_reports")
    parser.add_argument("--use-mock-llm", action="store_true",
                       help="Use mock LLM for testing (no API calls)")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing even if data exists")
    parser.add_argument("--clear-db", action="store_true",
                       help="Clear existing Neo4j database")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Skip visualization generation")

    args = parser.parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Smart Financial Knowledge Graph Pipeline             ║
║           With Automatic Duplicate Detection                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    # Determine input path
    if args.use_sample:
        input_path = None
        documents_to_check = list(SAMPLE_REPORTS_DIR.glob("*.pdf"))
    else:
        input_path = Path(args.input) if args.input else SAMPLE_REPORTS_DIR
        if input_path.is_file():
            documents_to_check = [input_path]
        else:
            documents_to_check = list(input_path.glob("*.pdf"))

    # Smart processing detection
    skip_extraction = False
    
    if not args.force_reprocess and not args.clear_db:
        print(f"\n{Fore.YELLOW}[Smart Check] Checking for existing processed data...{Style.RESET_ALL}")
        
        if documents_to_check:
            doc_names = [doc.name for doc in documents_to_check]
            print(f"Documents to process: {doc_names}")
            
            # Check if any of these documents appear to be processed
            for doc_name in doc_names:
                if check_document_processed(doc_name):
                    print(f"\n{Fore.GREEN}✓ Document '{doc_name}' appears to already be processed!")
                    print(f"  Use --force-reprocess to reprocess anyway")
                    print(f"  Use --clear-db to start fresh")
                    
                    skip_extraction = True
                    break
    
    if skip_extraction and not args.force_reprocess:
        print(f"\n{Fore.BLUE}[Skip Extraction] Using existing data in Neo4j{Style.RESET_ALL}")
        print(f"[Jump to Query Interface] Starting interactive queries...")
        
        # Connect to Neo4j and start query interface
        try:
            neo4j_client = Neo4jClient()
            neo4j_client.connect()
            
            # Skip to Step 5: Query Interface
            step_5_query_interface(neo4j_client)
            
        except Exception as e:
            print(f"{Fore.RED}Error connecting to Neo4j: {e}{Style.RESET_ALL}")
            print(f"Try running with --force-reprocess to rebuild the database")
            sys.exit(1)
        
        return

    # Original pipeline flow if processing is needed
    try:
        print(f"\n{Fore.YELLOW}[Step 1/5] Document Preprocessing{Style.RESET_ALL}")
        print("-" * 60)
        # Handle None input_path case
        actual_input_path = input_path if input_path is not None else SAMPLE_REPORTS_DIR
        documents = step_1_preprocessing(actual_input_path, use_sample=args.use_sample)
        
        print(f"\n{Fore.YELLOW}[Step 2/5] Entity & Relationship Extraction{Style.RESET_ALL}")
        print("-" * 60)
        if args.use_mock_llm:
            print(f"{Fore.BLUE}Using Mock LLM (no API calls){Style.RESET_ALL}")
        else:
            print(f"{Fore.BLUE}Using OpenRouter API with rate limiting{Style.RESET_ALL}")
            
        extraction = step_2_extraction(documents, use_mock=args.use_mock_llm, rate_limit_mode=True)
        
        print(f"\n{Fore.YELLOW}[Step 3/5] Graph Construction{Style.RESET_ALL}")
        print("-" * 60)
        neo4j_client, stats = step_3_graph_construction(extraction, clear_db=args.clear_db)
        
        if neo4j_client is None:
            print(f"{Fore.RED}Cannot proceed without Neo4j connection{Style.RESET_ALL}")
            sys.exit(1)
        
        print(f"\n{Fore.YELLOW}[Step 4/5] Visualization{Style.RESET_ALL}")
        print("-" * 60)
        if not args.skip_visualization:
            viz_files = step_4_visualization(neo4j_client)
            print(f"\n{Fore.YELLOW}Generated visualizations:{Style.RESET_ALL}")
            for viz_file in viz_files:
                print(f"  📊 {viz_file}")
        else:
            print(f"{Fore.BLUE}Skipped visualization generation{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}[Step 5/5] Query Interface{Style.RESET_ALL}")
        print("-" * 60)
        step_5_query_interface(neo4j_client)
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Pipeline interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Pipeline failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()