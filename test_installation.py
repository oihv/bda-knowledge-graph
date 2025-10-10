"""
Installation and System Test Script
Run this to verify everything is set up correctly
"""
import sys
import subprocess
from pathlib import Path
from colorama import init, Fore, Style

init()

def print_header(text):
    """Print formatted header"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

def print_success(text):
    """Print success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message"""
    print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

def test_python_version():
    """Test Python version"""
    print_header("Testing Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} (✓ >= 3.8)")
        return True
    else:
        print_error(f"Python {version_str} (✗ Need >= 3.8)")
        return False

def test_imports():
    """Test required package imports"""
    print_header("Testing Package Imports")
    
    required_packages = [
        ('dotenv', 'python-dotenv'),
        ('requests', 'requests'),
        ('fitz', 'PyMuPDF'),
        ('pdfplumber', 'pdfplumber'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('neo4j', 'neo4j'),
        ('networkx', 'networkx'),
        ('plotly', 'plotly'),
        ('pyvis', 'pyvis'),
        ('streamlit', 'streamlit'),
        ('tqdm', 'tqdm'),
        ('colorama', 'colorama'),
    ]
    
    all_success = True
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print_success(f"{package_name}")
        except ImportError:
            print_error(f"{package_name} (not installed)")
            all_success = False
    
    return all_success

def test_project_structure():
    """Test project directory structure"""
    print_header("Testing Project Structure")
    
    required_dirs = [
        'src',
        'src/preprocessing',
        'src/llm_extraction',
        'src/graph_builder',
        'src/visualization',
        'src/query_interface',
        'data',
        'data/sample_reports',
        'outputs',
        'outputs/visualizations',
    ]
    
    required_files = [
        'config.py',
        'main.py',
        'requirements.txt',
        '.env.example',
        'README.md',
    ]
    
    all_success = True
    
    print("Checking directories:")
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print_success(f"{dir_path}/")
        else:
            print_error(f"{dir_path}/ (missing)")
            all_success = False
    
    print("\nChecking files:")
    for file_path in required_files:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} (missing)")
            all_success = False
    
    return all_success

def test_env_config():
    """Test environment configuration"""
    print_header("Testing Environment Configuration")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print_warning(".env file not found")
        print("  ℹ️  Run: cp .env.example .env")
        print("  ℹ️  Then edit .env with your credentials")
        return False
    
    print_success(".env file exists")
    
    # Check if config can be imported
    try:
        import config
        print_success("config.py imports successfully")
        
        # Check critical config values
        if hasattr(config, 'NEO4J_URI'):
            print_success(f"NEO4J_URI: {config.NEO4J_URI}")
        
        if hasattr(config, 'OPENROUTER_API_KEY'):
            if config.OPENROUTER_API_KEY:
                print_success("OPENROUTER_API_KEY: Set (hidden)")
            else:
                print_warning("OPENROUTER_API_KEY: Not set (will use mock LLM)")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to import config: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j database connection"""
    print_header("Testing Neo4j Connection")
    
    try:
        from src.graph_builder.neo4j_client import Neo4jClient
        import config
        
        print(f"Attempting connection to: {config.NEO4J_URI}")
        
        client = Neo4jClient()
        client.connect()
        
        # Get statistics
        stats = client.get_graph_statistics()
        
        print_success("Connected to Neo4j!")
        print(f"  ℹ️  Current nodes: {stats['total_nodes']}")
        print(f"  ℹ️  Current relationships: {stats['total_relationships']}")
        
        client.close()
        return True
        
    except ImportError as e:
        print_error(f"Import error: {e}")
        return False
    except Exception as e:
        print_error(f"Connection failed: {e}")
        print("\n  Troubleshooting:")
        print("  1. Make sure Neo4j is running")
        print("  2. Check credentials in .env")
        print("  3. Verify URI (bolt://localhost:7687)")
        print("  4. Try: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
        return False

def test_module_imports():
    """Test project module imports"""
    print_header("Testing Project Module Imports")
    
    modules = [
        'src.preprocessing.pdf_processor',
        'src.preprocessing.text_cleaner',
        'src.llm_extraction.llm_client',
        'src.llm_extraction.entity_extractor',
        'src.graph_builder.neo4j_client',
        'src.graph_builder.graph_constructor',
        'src.visualization.graph_visualizer',
        'src.query_interface.nl_to_cypher',
    ]
    
    all_success = True
    
    for module in modules:
        try:
            __import__(module)
            print_success(module)
        except Exception as e:
            print_error(f"{module}: {e}")
            all_success = False
    
    return all_success

def test_sample_data():
    """Test sample data generation"""
    print_header("Testing Sample Data Generation")
    
    try:
        from src.preprocessing.pdf_processor import create_sample_text_documents
        
        documents = create_sample_text_documents()
        
        print_success(f"Generated {len(documents)} sample documents")
        
        for doc in documents:
            print(f"  ℹ️  {doc['filename']}: {len(doc['text'])} characters")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to generate sample data: {e}")
        return False

def test_mock_extraction():
    """Test mock LLM extraction"""
    print_header("Testing Mock LLM Extraction")
    
    try:
        from src.llm_extraction.llm_client import MockLLMClient
        from src.llm_extraction.entity_extractor import EntityExtractor
        from src.preprocessing.pdf_processor import create_sample_text_documents
        from src.preprocessing.text_cleaner import TextCleaner
        
        # Get sample data
        documents = create_sample_text_documents()
        doc = documents[0]
        
        # Preprocess
        processed = TextCleaner.preprocess_document(doc, chunk_size=1000, overlap=100)
        
        # Extract with mock LLM
        client = MockLLMClient()
        extractor = EntityExtractor(client)
        
        # Extract from first chunk
        extraction = extractor.extract_from_text(processed['chunks'][0])
        
        if extraction:
            print_success("Mock extraction successful")
            print(f"  ℹ️  Entities: {len(extraction['entities'])}")
            print(f"  ℹ️  Relationships: {len(extraction['relationships'])}")
            return True
        else:
            print_error("Extraction returned None")
            return False
        
    except Exception as e:
        print_error(f"Extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_pipeline_test():
    """Run a quick end-to-end pipeline test"""
    print_header("Running Quick Pipeline Test")
    
    try:
        print("This will test the full pipeline with sample data...")
        print("(This may take 30-60 seconds)")
        
        from src.preprocessing.pdf_processor import create_sample_text_documents
        from src.preprocessing.text_cleaner import TextCleaner
        from src.llm_extraction.llm_client import MockLLMClient
        from src.llm_extraction.entity_extractor import EntityExtractor
        
        # Step 1: Get sample data
        print("\n1️⃣  Loading sample documents...")
        documents = create_sample_text_documents()
        print_success(f"Loaded {len(documents)} documents")
        
        # Step 2: Preprocess
        print("\n2️⃣  Preprocessing...")
        preprocessed = []
        for doc in documents[:1]:  # Just test with one document
            processed = TextCleaner.preprocess_document(doc, chunk_size=1500, overlap=150)
            preprocessed.append(processed)
        print_success(f"Created {sum(d['num_chunks'] for d in preprocessed)} chunks")
        
        # Step 3: Extract
        print("\n3️⃣  Extracting entities (mock LLM)...")
        client = MockLLMClient()
        extractor = EntityExtractor(client)
        extractions = extractor.extract_from_documents(preprocessed)
        global_extraction = extractor.create_global_extraction(extractions)
        print_success(f"Extracted {len(global_extraction['entities'])} entities")
        print_success(f"Extracted {len(global_extraction['relationships'])} relationships")
        
        # Step 4: Test visualization (without Neo4j)
        print("\n4️⃣  Testing visualization components...")
        from src.visualization.graph_visualizer import GraphVisualizer
        print_success("Visualization modules loaded")
        
        print_success("\n🎉 Pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(results):
    """Print test summary"""
    print_header("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"{Fore.GREEN}Passed: {passed_tests}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {failed_tests}{Style.RESET_ALL}")
    
    print("\n" + "="*70)
    
    if failed_tests == 0:
        print(f"{Fore.GREEN}✅ All tests passed! System is ready.{Style.RESET_ALL}")
        print("\nNext steps:")
        print("  1. Run the pipeline: uv run python main.py --use-sample --use-mock-llm")
        print("  2. Open visualizations in: outputs/visualizations/")
        print("  3. Try Streamlit UI: uv run streamlit run streamlit_app.py")
    elif 'neo4j_connection' not in [k for k, v in results.items() if not v]:
        print(f"{Fore.YELLOW}⚠️  Most tests passed, but Neo4j is not connected.{Style.RESET_ALL}")
        print("\nYou can still:")
        print("  - Test extraction: Run with --use-mock-llm flag")
        print("  - Fix Neo4j: See QUICKSTART.md for setup instructions")
    else:
        print(f"{Fore.RED}❌ Some tests failed. Please fix issues above.{Style.RESET_ALL}")
        print("\nQuick fixes:")
        print("  - Missing packages: uv sync")
        print("  - Missing .env: cp .env.example .env")
        print("  - Neo4j issues: See QUICKSTART.md")
    
    print("="*70 + "\n")

def main():
    """Run all tests"""
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Financial Knowledge Graph - Installation Test        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
    """)
    
    results = {}
    
    # Run all tests
    results['python_version'] = test_python_version()
    results['imports'] = test_imports()
    results['project_structure'] = test_project_structure()
    results['env_config'] = test_env_config()
    results['module_imports'] = test_module_imports()
    results['sample_data'] = test_sample_data()
    results['mock_extraction'] = test_mock_extraction()
    results['neo4j_connection'] = test_neo4j_connection()
    
    # Optional: Full pipeline test
    print("\n" + "="*70)
    response = input("Run full pipeline test? (y/n): ").lower()
    if response == 'y':
        results['pipeline_test'] = run_quick_pipeline_test()
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()