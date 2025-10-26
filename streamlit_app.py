"""
Streamlit Web Interface for Financial Knowledge Graph System
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import json
from pathlib import Path
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from config import (OUTPUT_DIR, VISUALIZATION_DIR, SAMPLE_REPORTS_DIR,
                  CHUNK_SIZE, CHUNK_OVERLAP)
from src.preprocessing.pdf_processor import PDFProcessor, create_sample_text_documents
from src.preprocessing.text_cleaner import TextCleaner
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.llm_extraction.hf_llm_client import HuggingFaceLLMClient, HF_TEXT_GENERATION_MODELS, HF_MODEL_DESCRIPTIONS
from src.llm_extraction.entity_extractor import EntityExtractor
from src.graph_builder.neo4j_client import Neo4jClient
from src.graph_builder.graph_constructor import GraphConstructor
from src.visualization.graph_visualizer import GraphVisualizer
from src.query_interface.nl_to_cypher import NLToCypherConverter, QueryInterface

# Page configuration
st.set_page_config(
    page_title="Financial Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .existing-data-alert {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_existing_data(neo4j_client):
    """
    Check if Neo4j already contains processed data
    Returns: (has_data: bool, stats: dict)
    """
    try:
        stats = neo4j_client.get_graph_statistics()
        total_entities = sum(stats.get('node_types', {}).values())
        total_relationships = sum(stats.get('relationship_types', {}).values())
        
        # If we have substantial data, assume documents are processed
        has_data = total_entities > 50 and total_relationships > 50
        
        return has_data, {
            'total_entities': total_entities,
            'total_relationships': total_relationships,
            'node_types': stats.get('node_types', {}),
            'relationship_types': stats.get('relationship_types', {})
        }
        
    except Exception as e:
        st.warning(f"Could not check for existing data: {e}")
        return False, None

# Initialize session state
if 'neo4j_connected' not in st.session_state:
    st.session_state.neo4j_connected = False
if 'extraction_complete' not in st.session_state:
    st.session_state.extraction_complete = False
if 'graph_built' not in st.session_state:
    st.session_state.graph_built = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'existing_data_detected' not in st.session_state:
    st.session_state.existing_data_detected = False
if 'existing_data_stats' not in st.session_state:
    st.session_state.existing_data_stats = None

# Header
st.markdown('<div class="main-header">🕸️ Financial Knowledge Graph</div>', unsafe_allow_html=True)
st.markdown("### AI-Powered Document Analysis & Visualization")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Neo4j connection
    st.subheader("Neo4j Database")
    neo4j_uri = st.text_input("URI", value="bolt://localhost:7687")
    neo4j_user = st.text_input("Username", value="neo4j")
    neo4j_password = st.text_input("Password", type="password",
                                   value="password123")
    
    if st.button("🔌 Connect to Neo4j"):
        try:
            with st.spinner("Connecting..."):
                client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
                client.connect()
                st.session_state.neo4j_client = client
                st.session_state.neo4j_connected = True
                
                # Check for existing data
                has_data, stats = check_existing_data(client)
                st.session_state.existing_data_detected = has_data
                st.session_state.existing_data_stats = stats
                
                if has_data:
                    st.session_state.extraction_complete = True
                    st.session_state.graph_built = True
                
                st.success("✅ Connected!")
                
                if has_data and stats:
                    st.info(f"🔍 Detected existing data: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
        except Exception as e:
            st.error(f"❌ Failed: {e}")
    
    st.divider()
    
    # LLM configuration
    st.subheader("LLM Settings")
    use_mock = st.checkbox("Use Mock LLM (no API)", value=False)
    
    # Initialize api_key variables - check environment first
    import os
    env_api_key = os.getenv('OPENROUTER_API_KEY', '')
    env_hf_api_key = os.getenv('HUGGINGFACE_API_KEY', '') or os.getenv('HF_TOKEN', '')
    
    if not use_mock:
        api_key = st.text_input(
            "OpenRouter API Key", 
            type="password", 
            value=env_api_key,
            help="API calls for extraction are very cheap (~$0.001 per query). Leave empty to use environment variable."
        )
        model = st.selectbox(
            "Model",
            ["google/gemma-2-9b-it:free", 
             "meta-llama/llama-3.1-8b-instruct:free",
             "mistralai/mistral-7b-instruct:free",
             "deepseek/deepseek-r1-0528:free",
             "qwen/qwen3-235b-a22b:free"],
            help="Free models are available for extraction"
        )
        
        if api_key or env_api_key:
            st.success("✅ OpenRouter API Key available" + (" (from environment)" if env_api_key and not api_key else ""))
        else:
            st.info("🔑 Add OPENROUTER_API_KEY to your .env file or enter above")
    else:
        api_key = ""
        st.warning("⚠️ Mock LLM may generate invalid queries - use real API for best results")
    
    st.divider()
    
    # Query interface LLM provider selection
    st.subheader("Query Interface Settings")
    query_provider = st.radio(
        "LLM Provider for Queries",
        ["OpenRouter", "Hugging Face", "Mock"],
        help="Choose LLM provider for natural language to Cypher translation"
    )
    
    # Initialize query provider variables
    query_api_key = ""
    hf_api_key = ""
    hf_model = "deepseek-ai/DeepSeek-V3.1-Terminus"
    
    if query_provider == "OpenRouter":
        query_api_key = st.text_input(
            "OpenRouter API Key (Queries)", 
            type="password", 
            value=env_api_key,
            help="Very cheap for query translation (~$0.001 per query)"
        )
        query_model = st.selectbox(
            "Query Model",
            ["google/gemma-2-9b-it:free", 
             "meta-llama/llama-3.1-8b-instruct:free",
             "mistralai/mistral-7b-instruct:free"],
            help="Free models work well for query translation"
        )
        
        if query_api_key or env_api_key:
            st.success("✅ OpenRouter ready for queries" + (" (from environment)" if env_api_key and not query_api_key else ""))
        else:
            st.info("💡 Template matching works without API calls")
            st.info("🔑 Add OPENROUTER_API_KEY for better query translation")
            
    elif query_provider == "Hugging Face":
        hf_api_key = st.text_input(
            "Hugging Face API Key", 
            type="password", 
            value=env_hf_api_key,
            help="Free router API with rate limits. Get key at https://huggingface.co/settings/tokens"
        )
        hf_model = st.selectbox(
            "HF Model",
            list(HF_MODEL_DESCRIPTIONS.keys()),
            format_func=lambda x: HF_MODEL_DESCRIPTIONS[x],
            help="Choose model for query translation"
        )
        
        if hf_api_key or env_hf_api_key:
            st.success("✅ Hugging Face ready for queries" + (" (from environment)" if env_hf_api_key and not hf_api_key else ""))
        else:
            st.info("💡 Template matching works without API calls")
            st.info("🔑 Add HUGGINGFACE_API_KEY or HF_TOKEN to .env or enter above")
            
    else:  # Mock
        st.warning("⚠️ Mock LLM may generate invalid Cypher queries")
    
    # Processing settings
    st.subheader("Processing Settings")
    max_chunks = st.number_input("Max Chunks per Document", min_value=1, max_value=500, value=50)
    delay = st.slider("Delay per Chunk (seconds)", 1.0, 10.0, 3.0, 0.5)
    
    st.divider()
    
    # System status
    st.subheader("System Status")
    status_neo4j = "🟢 Connected" if st.session_state.neo4j_connected else "🔴 Disconnected"
    status_extraction = "✅ Complete" if st.session_state.extraction_complete else "⏳ Pending"
    status_graph = "✅ Built" if st.session_state.graph_built else "⏳ Pending"
    
    st.write(f"**Neo4j:** {status_neo4j}")
    st.write(f"**Extraction:** {status_extraction}")
    st.write(f"**Graph:** {status_graph}")
    
    if st.session_state.existing_data_detected and st.session_state.existing_data_stats:
        st.write("**Data Source:** 🔄 Existing")
        with st.expander("📊 Existing Data Details"):
            stats = st.session_state.existing_data_stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Entities", stats['total_entities'])
            with col2:
                st.metric("Relationships", stats['total_relationships'])
            
            st.write("**Node Types:**")
            for node_type, count in stats.get('node_types', {}).items():
                st.write(f"- {node_type}: {count}")
    elif st.session_state.extraction_complete:
        st.write("**Data Source:** 🆕 Fresh extraction")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Extract", "🕸️ Build Graph", "📊 Visualize", "💬 Query"])

# Tab 1: Upload and Extract
with tab1:
    st.header("Document Upload & Entity Extraction")
    
    # Show existing data alert if detected
    if st.session_state.existing_data_detected and st.session_state.existing_data_stats:
        st.markdown("""
        <div class="existing-data-alert">
            <h4>🔍 Existing Data Detected</h4>
            <p>Your Neo4j database already contains processed data. You can:</p>
            <ul>
                <li><strong>Skip to visualization/querying</strong> - Use the existing data</li>
                <li><strong>Force re-extraction</strong> - Process documents again (will overwrite)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Use Existing Data", type="primary", use_container_width=True):
                st.success("✅ Using existing data! Switch to 'Visualize' or 'Query' tabs.")
                st.balloons()
        with col2:
            force_reprocess = st.button("🔄 Force Re-extraction", use_container_width=True)
            if force_reprocess:
                st.session_state.existing_data_detected = False
                st.session_state.extraction_complete = False
                st.session_state.graph_built = False
                st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Documents")
        
        option = st.radio(
            "Choose input method:",
            ["Use Sample Documents", "Upload PDF Files"],
            disabled=st.session_state.processing or st.session_state.existing_data_detected
        )
        
        if option == "Upload PDF Files":
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                disabled=st.session_state.processing or st.session_state.existing_data_detected
            )
        else:
            uploaded_files = None
    
    with col2:
        st.subheader("Settings")
        st.info(f"Max Chunks: {max_chunks}")
        st.info(f"Delay: {delay}s")
        
        if st.session_state.existing_data_detected:
            st.success("🔍 Using Existing Data")
            if st.button("🔄 Reset to Fresh"):
                st.session_state.existing_data_detected = False
                st.session_state.extraction_complete = False
                st.session_state.graph_built = False
                st.rerun()
        elif st.session_state.extraction_complete:
            st.success("✅ Extraction Complete")
            if st.button("🔄 Reset"):
                st.session_state.extraction_complete = False
                st.session_state.graph_built = False
                st.rerun()
    
    extraction_disabled = (st.session_state.processing or 
                           st.session_state.existing_data_detected)
    
    if st.button("🚀 Start Extraction", 
                 type="primary", 
                 use_container_width=True,
                 disabled=extraction_disabled):
        
        st.session_state.processing = True
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Preprocessing
            status_text.text("📄 Loading documents...")
            progress_bar.progress(10)
            
            if option == "Use Sample Documents":
                documents = create_sample_text_documents()
            else:
                if not uploaded_files:
                    st.error("Please upload PDF files")
                    st.session_state.processing = False
                    st.stop()
                
                documents = []
                processor = PDFProcessor()
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    text = processor.extract_text(Path(tmp_path))
                    documents.append({
                        'filename': uploaded_file.name,
                        'text': text,
                        'metadata': {}
                    })
                    os.unlink(tmp_path)
            
            status_text.text(f"🧹 Preprocessing {len(documents)} documents...")
            progress_bar.progress(20)
            
            preprocessed = []
            for doc in documents:
                processed = TextCleaner.preprocess_document(doc)  # Using CHUNK_SIZE and CHUNK_OVERLAP from config
                # Limit chunks
                processed['chunks'] = processed['chunks'][:max_chunks]
                processed['num_chunks'] = len(processed['chunks'])
                preprocessed.append(processed)
            
            st.session_state.preprocessed_docs = preprocessed
            
            # Step 2: Extraction
            status_text.text("🤖 Extracting entities (this may take a while)...")
            progress_bar.progress(30)
            
            if use_mock:
                llm_client = MockLLMClient()
            else:
                # Check if we have an API key from input or environment
                effective_api_key = api_key or env_api_key
                if not effective_api_key:
                    st.error("❌ Please provide OpenRouter API Key in the sidebar or .env file")
                    st.session_state.processing = False
                    st.stop()
                
                # Set environment variable for LLMClient
                import os
                os.environ['OPENROUTER_API_KEY'] = effective_api_key
                llm_client = LLMClient()
            
            extractor = EntityExtractor(llm_client)
            
            all_extractions = []
            total_chunks = sum(d['num_chunks'] for d in preprocessed)
            processed_chunks = 0
            
            for doc_idx, doc in enumerate(preprocessed):
                doc_extractions = []
                
                for chunk_idx, chunk in enumerate(doc['chunks']):
                    try:
                        extraction = extractor.extract_from_text(chunk)
                        if extraction:
                            doc_extractions.append(extraction)
                        
                        processed_chunks += 1
                        progress = 30 + int((processed_chunks / total_chunks) * 50)
                        progress_bar.progress(min(progress, 80))
                        status_text.text(f"Processing chunk {processed_chunks}/{total_chunks}...")
                        
                        # Add delay
                        if not use_mock and chunk_idx < len(doc['chunks']) - 1:
                            import time
                            time.sleep(delay)
                            
                    except Exception as e:
                        st.warning(f"Error in chunk {chunk_idx + 1}: {str(e)}")
                        continue
                
                merged_doc = extractor.merge_extractions(doc_extractions)
                all_extractions.append(merged_doc)
            
            global_extraction = extractor.create_global_extraction(all_extractions)
            
            st.session_state.extraction = global_extraction
            st.session_state.extraction_complete = True
            
            progress_bar.progress(100)
            status_text.text("✅ Extraction complete!")
            
            # Display results
            st.success("🎉 Extraction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents", len(documents))
            with col2:
                st.metric("Entities", len(global_extraction['entities']))
            with col3:
                st.metric("Relationships", len(global_extraction['relationships']))
            
            # Show sample entities
            with st.expander("📋 View Extracted Entities (sample)"):
                import pandas as pd
                entity_data = []
                for entity in global_extraction['entities'][:20]:
                    entity_data.append({
                        'Name': entity['name'],
                        'Type': entity['type']
                    })
                st.dataframe(pd.DataFrame(entity_data), use_container_width=True)
            
            # Show sample relationships
            with st.expander("🔗 View Relationships (sample)"):
                rel_data = []
                for rel in global_extraction['relationships'][:20]:
                    rel_data.append({
                        'Source': rel['source'],
                        'Relationship': rel['type'],
                        'Target': rel['target']
                    })
                st.dataframe(pd.DataFrame(rel_data), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during extraction: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            st.session_state.processing = False

# Tab 2: Build Graph
with tab2:
    st.header("Knowledge Graph Construction")
    
    if not st.session_state.neo4j_connected:
        st.warning("⚠️ Please connect to Neo4j in the sidebar first")
    elif not st.session_state.extraction_complete:
        if st.session_state.existing_data_detected:
            st.info("ℹ️ Using existing graph data from Neo4j")
        else:
            st.warning("⚠️ Please complete extraction first")
    
    if st.session_state.neo4j_connected and st.session_state.extraction_complete:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Build Options")
            if st.session_state.existing_data_detected:
                st.info("Graph already exists in Neo4j")
                clear_db = st.checkbox("Clear existing database", value=False)
            else:
                clear_db = st.checkbox("Clear existing database", value=True)
            
            if st.button("🏗️ Build Graph", type="primary", use_container_width=True):
                with st.spinner("Building knowledge graph..."):
                    try:
                        constructor = GraphConstructor(st.session_state.neo4j_client)
                        stats = constructor.build_graph_from_extraction(
                            st.session_state.extraction,
                            clear_existing=clear_db
                        )
                        
                        st.session_state.graph_built = True
                        
                        st.success("✅ Graph Built Successfully!")
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Nodes Created", stats['nodes_created'])
                        with col2:
                            st.metric("Relationships Created", stats['relationships_created'])
                        with col3:
                            st.metric("Nodes Failed", stats['nodes_failed'])
                        with col4:
                            st.metric("Relationships Failed", stats['relationships_failed'])
                        
                    except Exception as e:
                        st.error(f"Error building graph: {e}")
        
        with col2:
            st.subheader("Graph Statistics")
            if st.session_state.graph_built:
                try:
                    stats = st.session_state.neo4j_client.get_graph_statistics()
                    st.metric("Total Nodes", stats['total_nodes'])
                    st.metric("Total Relationships", stats['total_relationships'])
                    
                    with st.expander("Node Types"):
                        for node_type, count in stats['node_types'].items():
                            st.write(f"**{node_type}:** {count}")
                except:
                    st.info("Run graph construction to see statistics")
        
        # NEW: Advanced Graph Cleanup Section
        if st.session_state.graph_built:
            st.divider()
            st.subheader("🔧 Advanced Graph Cleanup")
            
            st.info("💡 **Tip:** These tools help improve graph quality by removing noise and merging duplicates")
            
            # Create three columns for cleanup operations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Remove Abstract Nodes**")
                st.caption("Remove non-entity concepts like 'revenue', 'growth', 'the company'")
                if st.button("🧹 Clean Abstract Nodes", use_container_width=True):
                    with st.spinner("Removing abstract concept nodes..."):
                        try:
                            constructor = GraphConstructor(st.session_state.neo4j_client)
                            removed = constructor.cleanup_abstract_nodes()
                            if removed > 0:
                                st.success(f"✅ Removed {removed} abstract nodes")
                                st.balloons()
                            else:
                                st.info("✓ No abstract nodes found")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                st.markdown("**Merge Duplicate Nodes**")
                st.caption("Automatically merge nodes with overlapping original names")
                if st.button("🔗 Merge Duplicates", use_container_width=True):
                    with st.spinner("Merging duplicate nodes..."):
                        try:
                            constructor = GraphConstructor(st.session_state.neo4j_client)
                            merged = constructor.merge_similar_nodes()
                            if merged > 0:
                                st.success(f"✅ Merged {merged} duplicate nodes")
                                st.balloons()
                            else:
                                st.info("✓ No duplicates found")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col3:
                st.markdown("**Validate Graph**")
                st.caption("Check for isolated nodes, duplicates, and quality issues")
                if st.button("🔍 Validate Graph", use_container_width=True):
                    with st.spinner("Validating graph structure..."):
                        try:
                            constructor = GraphConstructor(st.session_state.neo4j_client)
                            issues = constructor.validate_graph()
                            
                            # Display validation results
                            st.success("✅ Validation Complete")
                            
                            # Create metrics for issues found
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Isolated Nodes", len(issues.get('isolated_nodes', [])))
                            with metric_col2:
                                st.metric("Potential Duplicates", len(issues.get('duplicate_nodes', [])))
                            with metric_col3:
                                st.metric("Abstract Nodes", len(issues.get('abstract_nodes', [])))
                            
                            # Show detailed issues in expandable sections
                            if issues.get('isolated_nodes'):
                                with st.expander(f"⚠️ Isolated Nodes ({len(issues['isolated_nodes'])})"):
                                    for node in issues['isolated_nodes'][:10]:
                                        st.write(f"- **{node.get('name')}** ({node.get('label')})")
                                    if len(issues['isolated_nodes']) > 10:
                                        st.caption(f"...and {len(issues['isolated_nodes']) - 10} more")
                            
                            if issues.get('duplicate_nodes'):
                                with st.expander(f"⚠️ Potential Duplicates ({len(issues['duplicate_nodes'])})"):
                                    for dup in issues['duplicate_nodes'][:10]:
                                        st.write(f"- **{dup.get('name1')}** ↔️ **{dup.get('name2')}** ({dup.get('label')})")
                                    if len(issues['duplicate_nodes']) > 10:
                                        st.caption(f"...and {len(issues['duplicate_nodes']) - 10} more")
                            
                            if issues.get('abstract_nodes'):
                                with st.expander(f"⚠️ Abstract Concept Nodes ({len(issues['abstract_nodes'])})"):
                                    for node in issues['abstract_nodes']:
                                        st.write(f"- **{node.get('name')}** ({node.get('label')})")
                            
                            # If graph is clean, show success message
                            if (not issues.get('isolated_nodes') and 
                                not issues.get('duplicate_nodes') and 
                                not issues.get('abstract_nodes')):
                                st.success("🎉 Graph is clean! No issues found.")
                        
                        except Exception as e:
                            st.error(f"Error validating graph: {e}")
            
            # Add a "Run All Cleanup" button
            st.markdown("---")
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                if st.button("⚡ Run All Cleanup Operations", type="secondary", use_container_width=True):
                    with st.spinner("Running comprehensive graph cleanup..."):
                        try:
                            constructor = GraphConstructor(st.session_state.neo4j_client)
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Step 1: Remove abstract nodes
                            status_text.text("1/3 Removing abstract concept nodes...")
                            progress_bar.progress(33)
                            removed = constructor.cleanup_abstract_nodes()
                            
                            # Step 2: Merge duplicates
                            status_text.text("2/3 Merging duplicate nodes...")
                            progress_bar.progress(66)
                            merged = constructor.merge_similar_nodes()
                            
                            # Step 3: Validate
                            status_text.text("3/3 Validating final graph...")
                            progress_bar.progress(100)
                            issues = constructor.validate_graph()
                            
                            status_text.empty()
                            progress_bar.empty()
                            
                            # Show summary
                            st.success("✅ Comprehensive Cleanup Complete!")
                            
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            with summary_col1:
                                st.metric("Nodes Removed", removed)
                            with summary_col2:
                                st.metric("Nodes Merged", merged)
                            with summary_col3:
                                remaining_issues = (len(issues.get('isolated_nodes', [])) + 
                                                  len(issues.get('duplicate_nodes', [])) + 
                                                  len(issues.get('abstract_nodes', [])))
                                st.metric("Remaining Issues", remaining_issues)
                            
                            if remaining_issues == 0:
                                st.balloons()
                                st.success("🎉 Your graph is now optimized and clean!")
                            else:
                                st.info(f"💡 Graph improved! {remaining_issues} minor issues remain (check validation results above)")
                        
                        except Exception as e:
                            st.error(f"Error during cleanup: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

# Tab 3: Visualize
with tab3:
    st.header("Graph Visualization")
    
    if not st.session_state.graph_built:
        if st.session_state.existing_data_detected:
            st.info("ℹ️ Ready to visualize existing graph data")
        else:
            st.warning("⚠️ Please build the graph first")
    
    if st.session_state.graph_built:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Interactive Network (PyVis)", "Plotly Graph", "Statistics Dashboard"]
        )
        
        if st.button("📊 Generate Visualization", type="primary"):
            with st.spinner("Generating visualization..."):
                try:
                    visualizer = GraphVisualizer(st.session_state.neo4j_client)
                    
                    if viz_type == "Interactive Network (PyVis)":
                        file_path = visualizer.create_pyvis_network(output_file="streamlit_graph.html")
                        st.success(f"✅ Generated: {file_path}")
                        
                        # Read and display
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800, scrolling=True)
                    
                    elif viz_type == "Plotly Graph":
                        file_path = visualizer.create_plotly_network(output_file="streamlit_plotly.html")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800)
                    
                    else:  # Statistics
                        file_path = visualizer.create_statistics_visualization("streamlit_stats.html")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(str(e))

# Tab 4: Query
with tab4:
    st.header("Natural Language Query Interface")
    
    if not st.session_state.graph_built:
        if st.session_state.existing_data_detected:
            st.info("ℹ️ Ready to query existing graph data")
        else:
            st.warning("⚠️ Please build the graph first")
    
    if st.session_state.graph_built:
        # Initialize query interface with selected provider
        if 'query_interface' not in st.session_state or st.session_state.get('query_provider') != query_provider:
            st.session_state.query_provider = query_provider
            
            llm_client = None
            if query_provider == "Mock":
                llm_client = MockLLMClient()
            elif query_provider == "OpenRouter":
                # Check if we have an API key from input or environment
                effective_api_key = query_api_key or env_api_key
                if not effective_api_key:
                    st.error("❌ Please provide OpenRouter API Key in the sidebar or .env file to use query interface")
                    st.stop()
                
                # Set environment variable for LLMClient
                import os
                os.environ['OPENROUTER_API_KEY'] = effective_api_key
                llm_client = LLMClient()
                
            elif query_provider == "Hugging Face":
                # Check if we have HF API key
                effective_hf_key = hf_api_key or env_hf_api_key
                if not effective_hf_key:
                    st.error("❌ Please provide Hugging Face API Key in the sidebar or .env file to use query interface")
                    st.stop()
                
                llm_client = HuggingFaceLLMClient(api_key=effective_hf_key, model=hf_model)
            
            if llm_client:
                converter = NLToCypherConverter(llm_client, st.session_state.neo4j_client)
                st.session_state.query_interface = QueryInterface(converter)
        
        # Show current provider status
        if query_provider == "OpenRouter":
            if query_api_key or env_api_key:
                st.info(f"🤖 Using OpenRouter for query translation")
            else:
                st.warning("⚠️ OpenRouter API key missing - template matching only")
        elif query_provider == "Hugging Face":
            if hf_api_key or env_hf_api_key:
                model_name = hf_model.split('/')[-1] if hf_model else "unknown"
                st.info(f"🤗 Using Hugging Face ({model_name}) for query translation")
            else:
                st.warning("⚠️ Hugging Face API key missing - template matching only")
        else:
            st.warning("🔧 Using Mock LLM - results may be unreliable")
        
        # Suggested questions - make them more prominent
        st.subheader("💡 Quick Start Questions")
        
        # Show current query behavior
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("🤖 **All queries are processed by LLM for maximum flexibility and accuracy**")
        with col2:
            if st.button("🔧 Show All Suggestions", key="expand_suggestions"):
                st.session_state.show_all_suggestions = not st.session_state.get('show_all_suggestions', False)
        
        # Get suggestions and organize them by category
        suggestions = st.session_state.query_interface.converter.get_suggested_questions()
        
        # Show first 6 suggestions as prominent buttons
        st.markdown("**Most Popular:**")
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions[:6]):
            col_idx = i % 3
            with cols[col_idx]:
                if st.button(f"📊 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.suggested_query = suggestion
                    st.rerun()
        
        # Show remaining suggestions if expanded
        if st.session_state.get('show_all_suggestions', False):
            st.markdown("**More Options:**")
            remaining_suggestions = suggestions[6:]
            if remaining_suggestions:
                cols2 = st.columns(2)
                for i, suggestion in enumerate(remaining_suggestions):
                    col_idx = i % 2
                    with cols2[col_idx]:
                        if st.button(f"🔍 {suggestion}", key=f"suggestion_extra_{i}", use_container_width=True):
                            st.session_state.suggested_query = suggestion
                            st.rerun()
            
        # Auto-fill from suggestions
        default_query = st.session_state.get('suggested_query', '')
        if default_query:
            st.session_state.suggested_query = ''  # Clear after use
        
        # Query input
        question = st.text_input(
            "Ask a question about your knowledge graph:", 
            value=default_query,
            placeholder="e.g., List companies, Apple products, Count people"
        )
        
        if st.button("🔍 Search", type="primary"):
            if question:
                with st.spinner("Processing query..."):
                    try:
                        result = st.session_state.query_interface.process_query(question)
                        st.text_area("Results", result, height=400)
                    except Exception as e:
                        st.error(f"Error: {e}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    Financial Knowledge Graph System | Built with Streamlit & Neo4j
</div>
""", unsafe_allow_html=True)
