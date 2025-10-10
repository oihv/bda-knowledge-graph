#!/usr/bin/env python3
"""
Natural Language Query Tool
Interactive command-line interface for querying the knowledge graph with natural language
"""
import argparse
import sys
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.graph_builder.neo4j_client import Neo4jClient
from src.query_interface.nl_to_cypher import NLToCypherConverter, QueryInterface
import config


def print_banner():
    """Print application banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           Natural Language Knowledge Graph Query            ║
║              Ask questions in plain English!                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def setup_clients(use_mock_llm: bool = False):
    """
    Setup LLM and Neo4j clients
    
    Args:
        use_mock_llm: Whether to use mock LLM (for testing without API)
        
    Returns:
        Tuple of (llm_client, neo4j_client, converter, interface)
    """
    # Setup LLM client
    if use_mock_llm or not config.OPENROUTER_API_KEY:
        print(f"{Fore.YELLOW}⚠️  Using Mock LLM (no API calls){Style.RESET_ALL}")
        print("   To use real LLM, add OPENROUTER_API_KEY to your .env file")
        llm_client = MockLLMClient()
    else:
        print(f"{Fore.GREEN}✅ Using OpenRouter API{Style.RESET_ALL}")
        llm_client = LLMClient()
    
    # Setup Neo4j client
    try:
        neo4j_client = Neo4jClient(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
        neo4j_client.connect()
        print(f"{Fore.GREEN}✅ Connected to Neo4j at {config.NEO4J_URI}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to connect to Neo4j: {e}{Style.RESET_ALL}")
        print("   Make sure Neo4j is running and credentials are correct")
        return None, None, None, None
    
    # Create converter and interface
    converter = NLToCypherConverter(llm_client, neo4j_client)
    interface = QueryInterface(converter)
    
    return llm_client, neo4j_client, converter, interface


def interactive_mode(interface: QueryInterface):
    """
    Run interactive query mode
    
    Args:
        interface: Query interface instance
    """
    print(f"\n{Fore.GREEN}🚀 Interactive Query Mode{Style.RESET_ALL}")
    print("Type your questions in natural language. Special commands:")
    print("  • 'help' or 'suggestions' - Show suggested questions")
    print("  • 'history' - Show query history")
    print("  • 'quit' or 'exit' - Exit the program")
    print("-" * 60)
    
    # Show suggestions
    print(interface.show_suggestions())
    print()
    
    while True:
        try:
            # Get user input
            question = input(f"{Fore.BLUE}❓ Your question: {Style.RESET_ALL}").strip()
            
            if not question:
                continue
                
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print(f"{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
                break
                
            elif question.lower() in ['help', 'suggestions', 'h']:
                print(interface.show_suggestions())
                continue
                
            elif question.lower() in ['history', 'hist']:
                print(interface.show_history())
                continue
            
            # Process the query
            print(f"\n{Fore.YELLOW}🔄 Processing query...{Style.RESET_ALL}")
            result = interface.process_query(question)
            
            # Display results
            print(f"\n{Fore.GREEN}📊 Results:{Style.RESET_ALL}")
            print(result)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def batch_mode(interface: QueryInterface, questions: list):
    """
    Run batch query mode
    
    Args:
        interface: Query interface instance
        questions: List of questions to process
    """
    print(f"\n{Fore.GREEN}📋 Batch Query Mode{Style.RESET_ALL}")
    print(f"Processing {len(questions)} questions...")
    print("-" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{Fore.BLUE}❓ Question {i}: {question}{Style.RESET_ALL}")
        
        result = interface.process_query(question)
        print(f"\n{Fore.GREEN}📊 Results:{Style.RESET_ALL}")
        print(result)
        print("-" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Natural Language Query Tool for Knowledge Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_tool.py                              # Interactive mode
  python query_tool.py --mock                       # Interactive mode with mock LLM
  python query_tool.py --batch "What products does Apple make?" "Count all companies"
  python query_tool.py --suggestions                # Show suggested questions
        """
    )
    
    parser.add_argument(
        '--mock', 
        action='store_true',
        help='Use mock LLM client (no API calls)'
    )
    
    parser.add_argument(
        '--batch',
        nargs='+',
        help='Run specific questions in batch mode'
    )
    
    parser.add_argument(
        '--suggestions',
        action='store_true',
        help='Show suggested questions and exit'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Setup clients
    llm_client, neo4j_client, converter, interface = setup_clients(args.mock)
    
    if not interface:
        sys.exit(1)
    
    try:
        # Show suggestions and exit
        if args.suggestions:
            print(interface.show_suggestions())
            return
        
        # Batch mode
        if args.batch:
            batch_mode(interface, args.batch)
        else:
            # Interactive mode
            interactive_mode(interface)
    
    finally:
        # Cleanup
        if neo4j_client:
            neo4j_client.close()


if __name__ == "__main__":
    main()