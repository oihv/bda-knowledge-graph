"""
Main Pipeline Script
Execute the complete knowledge graph extraction and visualization pipeline
"""
import argparse
import json
import sys
from pathlib import Path
import logging
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Import project modules
from config import SAMPLE_REPORTS_DIR, OUTPUT_DIR, PROCESSED_DIR, OPENROUTER_API_KEY
from src.preprocessing.pdf_processor import PDFProcessor, create_sample_text_documents
from src.preprocessing.text_cleaner import TextCleaner
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.llm_extraction.entity_extractor import EntityExtractor
from src.graph_builder.neo4j_client import Neo4jClient
from src.graph_builder.graph_constructor import GraphConstructor
from src.visualization.graph_visualizer import GraphVisualizer
from src.query_interface.nl_to_cypher import NLToCypherConverter, QueryInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Financial Knowledge Graph Extraction System          ║
║           AI-Powered Document Analysis Pipeline             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def setup_directories():
    """Ensure all required directories exist"""
    dirs = [SAMPLE_REPORTS_DIR, OUTPUT_DIR, PROCESSED_DIR]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Directories initialized")


def step_1_preprocessing(input_path: Path, use_sample: bool = False):
    """
    Step 1: Preprocess documents (PDF extraction and cleaning)
    
    Args:
        input_path: Path to input documents
        use_sample: Whether to use sample documents
        
    Returns:
        List of preprocessed documents
    """
    print(f"\n{Fore.GREEN}[Step 1/5] Document Preprocessing{Style.RESET_ALL}")
    print("-" * 60)
    
    if use_sample:
        logger.info("Using sample text documents")
        documents = create_sample_text_documents()
    else:
        # Process PDFs from directory
        pdf_processor = PDFProcessor(method='pymupdf')
        documents = pdf_processor.process_directory(input_path)
    
    # Clean and chunk documents with rate limit-friendly settings
    preprocessed_docs = []
    for doc in documents:
        # Limit chunks to avoid rate limits - process first 50 chunks (100k characters)
        processed = TextCleaner.preprocess_document(doc, chunk_size=2000, overlap=200, max_chunks=50)
        preprocessed_docs.append(processed)
    
    # Save preprocessed data
    output_file = PROCESSED_DIR / "preprocessed_documents.json"
    with open(output_file, 'w') as f:
        json.dump(preprocessed_docs, f, indent=2)
    
    logger.info(f"Preprocessed {len(preprocessed_docs)} documents")
    logger.info(f"Saved to: {output_file}")
    
    return preprocessed_docs


def step_2_extraction(documents: list, use_mock: bool = False, rate_limit_mode: bool = True):
    """
    Step 2: Extract entities and relationships using LLM
    
    Args:
        documents: List of preprocessed documents
        use_mock: Whether to use mock LLM (for testing without API)
        rate_limit_mode: Whether to use rate limiting settings
        
    Returns:
        List of extraction results
    """
    print(f"\n{Fore.GREEN}[Step 2/5] Entity & Relationship Extraction{Style.RESET_ALL}")
    print("-" * 60)
    
    # Initialize LLM client
    if use_mock or not OPENROUTER_API_KEY:
        logger.warning("Using Mock LLM Client (no API calls)")
        llm_client = MockLLMClient()
    else:
        logger.info("Using OpenRouter API")
        llm_client = LLMClient()
    
    # Initialize extractor with rate limiting if enabled
    if rate_limit_mode:
        # Ultra-conservative settings for free tier APIs (hitting rate limits)
        extractor = EntityExtractor(llm_client, batch_size=1, delay_between_batches=600.0)
        logger.info("Using ultra-conservative rate limiting: 1 chunk per batch, 10 minute delays")
    else:
        # Faster processing for paid APIs
        extractor = EntityExtractor(llm_client, batch_size=20, delay_between_batches=10.0)
        logger.info("Using fast processing mode")
    
    # Extract from all documents
    extractions = extractor.extract_from_documents(documents)
    
    # Create global extraction
    global_extraction = extractor.create_global_extraction(extractions)
    
    # Save extractions
    output_file = OUTPUT_DIR / "extractions.json"
    with open(output_file, 'w') as f:
        json.dump(global_extraction, f, indent=2)
    
    logger.info(f"Extracted {len(global_extraction['entities'])} entities "
               f"and {len(global_extraction['relationships'])} relationships")
    logger.info(f"Saved to: {output_file}")
    
    return global_extraction


def step_3_graph_construction(extraction: dict, clear_db: bool = True):
    """
    Step 3: Build knowledge graph in Neo4j
    
    Args:
        extraction: Extraction results
        clear_db: Whether to clear existing database
        
    Returns:
        Construction statistics
    """
    print(f"\n{Fore.GREEN}[Step 3/5] Knowledge Graph Construction{Style.RESET_ALL}")
    print("-" * 60)
    
    try:
        # Connect to Neo4j
        neo4j_client = Neo4jClient()
        neo4j_client.connect()
        
        # Initialize constructor
        constructor = GraphConstructor(neo4j_client)
        
        # Build graph
        stats = constructor.build_graph_from_extraction(extraction, clear_existing=clear_db)
        
        logger.info(f"Graph construction complete:")
        logger.info(f"  - Nodes created: {stats['nodes_created']}")
        logger.info(f"  - Relationships created: {stats['relationships_created']}")
        
        # Get final statistics
        graph_stats = neo4j_client.get_graph_statistics()
        logger.info(f"Final graph statistics:")
        logger.info(f"  - Total nodes: {graph_stats['total_nodes']}")
        logger.info(f"  - Total relationships: {graph_stats['total_relationships']}")
        
        return neo4j_client, stats
    
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        logger.error("Make sure Neo4j is running and credentials are correct")
        return None, None


def step_4_visualization(neo4j_client: Neo4jClient):
    """
    Step 4: Create visualizations
    
    Args:
        neo4j_client: Connected Neo4j client
        
    Returns:
        List of generated visualization files
    """
    print(f"\n{Fore.GREEN}[Step 4/5] Graph Visualization{Style.RESET_ALL}")
    print("-" * 60)
    
    visualizer = GraphVisualizer(neo4j_client)
    
    generated_files = []
    
    try:
        # Create PyVis interactive visualization
        logger.info("Creating interactive PyVis visualization...")
        pyvis_file = visualizer.create_pyvis_network()
        generated_files.append(pyvis_file)
        
        # Create Plotly visualization
        logger.info("Creating Plotly visualization...")
        plotly_file = visualizer.create_plotly_network()
        generated_files.append(plotly_file)
        
        # Create statistics visualization
        logger.info("Creating statistics visualization...")
        stats_file = visualizer.create_statistics_visualization()
        generated_files.append(stats_file)
        
        # Export graph data
        logger.info("Exporting graph data...")
        json_file = visualizer.export_graph_data()
        generated_files.append(json_file)
        
        logger.info(f"Generated {len(generated_files)} visualization files")
        
        return generated_files
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return generated_files


def step_5_query_interface(neo4j_client: Neo4jClient, use_mock: bool = False):
    """
    Step 5: Interactive query interface
    
    Args:
        neo4j_client: Connected Neo4j client
        use_mock: Whether to use mock LLM
    """
    print(f"\n{Fore.GREEN}[Step 5/5] Natural Language Query Interface{Style.RESET_ALL}")
    print("-" * 60)
    
    # Initialize LLM client
    if use_mock or not OPENROUTER_API_KEY:
        llm_client = MockLLMClient()
    else:
        llm_client = LLMClient()
    
    # Initialize converter and interface
    converter = NLToCypherConverter(llm_client, neo4j_client)
    interface = QueryInterface(converter)
    
    # Show suggested questions
    print(interface.show_suggestions())
    
    # Interactive loop
    print(f"\n{Fore.YELLOW}Enter natural language questions (type 'exit' to quit):{Style.RESET_ALL}")
    
    while True:
        try:
            question = input(f"\n{Fore.CYAN}Query>{Style.RESET_ALL} ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if question.lower() == 'history':
                print(interface.show_history())
                continue
            
            if question.lower() == 'suggestions':
                print(interface.show_suggestions())
                continue
            
            if not question:
                continue
            
            # Process query
            result = interface.process_query(question)
            print(f"\n{result}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
    
    print(f"\n{Fore.YELLOW}Query interface closed.{Style.RESET_ALL}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Financial Knowledge Graph Extraction System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input directory containing PDF files'
    )
    
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Use sample text documents instead of PDFs'
    )
    
    parser.add_argument(
        '--use-mock-llm',
        action='store_true',
        help='Use mock LLM client (no API calls)'
    )
    
    parser.add_argument(
        '--clear-db',
        action='store_true',
        default=True,
        help='Clear existing Neo4j database'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip extraction step (load from previous run)'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )
    
    parser.add_argument(
        '--skip-query',
        action='store_true',
        help='Skip query interface'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup
    setup_directories()
    
    # Determine input path
    if args.use_sample:
        input_path = None
    else:
        input_path = Path(args.input) if args.input else SAMPLE_REPORTS_DIR
    
    try:
        # Step 1: Preprocessing
        documents = step_1_preprocessing(input_path, use_sample=args.use_sample)
        
        # Step 2: Extraction
        if args.skip_extraction:
            logger.info("Loading previous extraction results...")
            with open(OUTPUT_DIR / "extractions.json", 'r') as f:
                extraction = json.load(f)
        else:
            extraction = step_2_extraction(documents, use_mock=args.use_mock_llm, rate_limit_mode=True)
        
        # Step 3: Graph Construction
        neo4j_client, stats = step_3_graph_construction(extraction, clear_db=args.clear_db)
        
        if neo4j_client is None:
            logger.error("Cannot proceed without Neo4j connection")
            sys.exit(1)
        
        # Step 4: Visualization
        if not args.skip_visualization:
            viz_files = step_4_visualization(neo4j_client)
            print(f"\n{Fore.YELLOW}Generated visualizations:{Style.RESET_ALL}")
            for file in viz_files:
                print(f"  - {file}")
        
        # Step 5: Query Interface
        if not args.skip_query:
            step_5_query_interface(neo4j_client, use_mock=args.use_mock_llm)
        
        # Cleanup
        neo4j_client.close()
        
        # Final summary
        print(f"\n{Fore.GREEN}╔══════════════════════════════════════════════╗")
        print(f"║     Pipeline Execution Complete! ✓          ║")
        print(f"╚══════════════════════════════════════════════╝{Style.RESET_ALL}\n")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Pipeline interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()