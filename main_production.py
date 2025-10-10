"""
Production Pipeline for Large Document Processing
Handles 200+ chunks with resume capability and batch processing
"""
import argparse
import json
import sys
import time
from pathlib import Path
import logging
from colorama import init, Fore, Style
from datetime import datetime
from tqdm import tqdm

init()

from config import SAMPLE_REPORTS_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.text_cleaner import TextCleaner
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.llm_extraction.entity_extractor import EntityExtractor
from src.graph_builder.neo4j_client import Neo4jClient
from src.graph_builder.graph_constructor import GraphConstructor
from src.visualization.graph_visualizer import GraphVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionExtractor:
    """Production-ready extractor with checkpointing and resume"""
    
    def __init__(self, extractor: EntityExtractor, output_dir: Path, delay: float = 3.0):
        self.extractor = extractor
        self.output_dir = output_dir
        self.delay = delay
        self.checkpoint_file = output_dir / "checkpoint.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self) -> dict:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"processed_chunks": [], "extractions": [], "last_chunk_index": -1}
    
    def save_checkpoint(self, checkpoint: dict):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def extract_with_resume(self, document: dict, start_chunk: int = 0, 
                           max_chunks: int = None, batch_size: int = 50) -> dict:
        """
        Extract with checkpointing and resume capability
        
        Args:
            document: Preprocessed document
            start_chunk: Starting chunk index
            max_chunks: Maximum chunks to process (None = all)
            batch_size: Save checkpoint every N chunks
        """
        checkpoint = self.load_checkpoint()
        chunks = document.get('chunks', [])
        
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        # Resume from checkpoint
        start_idx = max(start_chunk, checkpoint['last_chunk_index'] + 1)
        extractions = checkpoint['extractions']
        
        logger.info(f"Processing {len(chunks)} chunks (starting from chunk {start_idx})")
        
        # Process chunks with progress bar
        for i in tqdm(range(start_idx, len(chunks)), 
                     desc="Extracting", 
                     initial=start_idx,
                     total=len(chunks)):
            
            # Add delay
            if i > start_idx:
                time.sleep(self.delay)
            
            chunk = chunks[i]
            
            try:
                extraction = self.extractor.extract_from_text(chunk)
                
                if extraction:
                    extractions.append(extraction)
                    logger.info(f"✓ Chunk {i+1}/{len(chunks)}: "
                              f"{len(extraction['entities'])} entities, "
                              f"{len(extraction['relationships'])} relationships")
                else:
                    logger.warning(f"✗ Chunk {i+1}/{len(chunks)}: No extraction")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue
            
            # Save checkpoint every batch_size chunks
            if (i + 1) % batch_size == 0:
                checkpoint['last_chunk_index'] = i
                checkpoint['extractions'] = extractions
                self.save_checkpoint(checkpoint)
                logger.info(f"💾 Checkpoint saved at chunk {i+1}")
        
        # Final checkpoint
        checkpoint['last_chunk_index'] = len(chunks) - 1
        checkpoint['extractions'] = extractions
        checkpoint['processed_chunks'] = list(range(len(chunks)))
        self.save_checkpoint(checkpoint)
        
        # Merge extractions
        merged = self.extractor.merge_extractions(extractions)
        merged['document_name'] = document.get('filename', 'Unknown')
        merged['chunks_processed'] = len(extractions)
        merged['total_chunks'] = len(chunks)
        merged['success_rate'] = len(extractions) / len(chunks) if chunks else 0
        
        return merged


def print_banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Production Knowledge Graph Extraction Pipeline       ║
║             Handles 200+ Chunks with Resume                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def main():
    parser = argparse.ArgumentParser(description='Production Pipeline for Large Documents')
    
    parser.add_argument('--input', type=str, required=True, 
                       help='Input PDF file or directory')
    parser.add_argument('--delay', type=float, default=3.0,
                       help='Delay between API calls (seconds)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Save checkpoint every N chunks')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--skip-graph', action='store_true',
                       help='Skip graph construction (extract only)')
    parser.add_argument('--use-mock', action='store_true',
                       help='Use mock LLM for testing')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Session directory: {session_dir}")
    
    # Initialize LLM
    if args.use_mock:
        logger.info("Using Mock LLM")
        llm_client = MockLLMClient()
    else:
        logger.info(f"Using OpenRouter API (delay: {args.delay}s, batch: {args.batch_size} chunks)")
        llm_client = LLMClient()
    
    extractor = EntityExtractor(llm_client)
    prod_extractor = ProductionExtractor(extractor, session_dir, delay=args.delay)
    
    # Step 1: Load and preprocess
    print(f"\n{Fore.GREEN}[Step 1/4] Document Preprocessing{Style.RESET_ALL}")
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text(input_path)
        documents = [{
            'filename': input_path.name,
            'text': text,
            'metadata': pdf_processor.extract_metadata(input_path)
        }]
    else:
        pdf_processor = PDFProcessor()
        documents = pdf_processor.process_directory(input_path)
    
    logger.info(f"Found {len(documents)} documents")
    
    preprocessed_docs = []
    for doc in documents:
        processed = TextCleaner.preprocess_document(doc, chunk_size=2000, overlap=200)
        preprocessed_docs.append(processed)
        logger.info(f"  {doc['filename']}: {processed['num_chunks']} chunks")
    
    # Step 2: Extract with checkpointing
    print(f"\n{Fore.GREEN}[Step 2/4] Entity Extraction (with checkpointing){Style.RESET_ALL}")
    
    if args.resume:
        logger.info("Resume mode enabled - loading from checkpoint")
    
    all_extractions = []
    
    for i, doc in enumerate(preprocessed_docs, 1):
        print(f"\n--- Document {i}/{len(preprocessed_docs)}: {doc['filename']} ---")
        logger.info(f"Total chunks: {doc['num_chunks']}")
        
        extraction = prod_extractor.extract_with_resume(
            doc,
            batch_size=args.batch_size
        )
        
        all_extractions.append(extraction)
        
        # Save document extraction
        doc_file = session_dir / f"extraction_doc_{i}.json"
        with open(doc_file, 'w') as f:
            json.dump(extraction, f, indent=2)
        
        logger.info(f"✅ Document {i} complete:")
        logger.info(f"   Entities: {len(extraction['entities'])}")
        logger.info(f"   Relationships: {len(extraction['relationships'])}")
        logger.info(f"   Success rate: {extraction['success_rate']:.1%}")
        logger.info(f"   Saved to: {doc_file}")
    
    # Merge all extractions
    global_extraction = extractor.create_global_extraction(all_extractions)
    
    # Save final result
    final_file = session_dir / "extraction_final.json"
    with open(final_file, 'w') as f:
        json.dump(global_extraction, f, indent=2)
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"Extraction Summary:")
    print(f"{'='*70}{Style.RESET_ALL}")
    print(f"Total Documents: {len(preprocessed_docs)}")
    print(f"Total Chunks: {sum(d['num_chunks'] for d in preprocessed_docs)}")
    print(f"Entities Extracted: {Fore.GREEN}{len(global_extraction['entities'])}{Style.RESET_ALL}")
    print(f"Relationships Found: {Fore.GREEN}{len(global_extraction['relationships'])}{Style.RESET_ALL}")
    print(f"Output: {final_file}")
    print(f"{'='*70}\n")
    
    if args.skip_graph:
        logger.info("Skipping graph construction (--skip-graph flag)")
        return
    
    # Step 3: Build graph
    print(f"\n{Fore.GREEN}[Step 3/4] Knowledge Graph Construction{Style.RESET_ALL}")
    
    try:
        neo4j_client = Neo4jClient()
        neo4j_client.connect()
        
        constructor = GraphConstructor(neo4j_client)
        stats = constructor.build_graph_from_extraction(global_extraction, clear_existing=True)
        
        logger.info(f"Graph built successfully:")
        logger.info(f"   Nodes: {stats['nodes_created']}")
        logger.info(f"   Relationships: {stats['relationships_created']}")
        
        # Step 4: Visualize
        print(f"\n{Fore.GREEN}[Step 4/4] Create Visualizations{Style.RESET_ALL}")
        
        visualizer = GraphVisualizer(neo4j_client)
        
        logger.info("Creating PyVis visualization...")
        viz_file = visualizer.create_pyvis_network(f"graph_{session_id}.html")
        logger.info(f"✅ Saved: {viz_file}")
        
        logger.info("Creating Plotly visualization...")
        plotly_file = visualizer.create_plotly_network(f"plotly_{session_id}.html")
        logger.info(f"✅ Saved: {plotly_file}")
        
        logger.info("Exporting graph data...")
        json_file = visualizer.export_graph_data(f"graph_data_{session_id}.json")
        logger.info(f"✅ Saved: {json_file}")
        
        neo4j_client.close()
        
        print(f"\n{Fore.GREEN}✅ Pipeline Complete!{Style.RESET_ALL}\n")
        print("View results:")
        print(f"   Neo4j Browser: http://localhost:7474")
        print(f"   Visualizations: {viz_file}")
        print(f"   Extracted Data: {final_file}\n")
        
    except Exception as e:
        logger.error(f"Graph construction failed: {e}")
        logger.info(f"Extracted data is still available at: {final_file}")


if __name__ == "__main__":
    main()