"""
Optimized Main Pipeline for Rate-Limited APIs
Processes documents with delays and batch management
"""
import argparse
import json
import sys
import time
from pathlib import Path
import logging
from colorama import init, Fore, Style

init()

from config import SAMPLE_REPORTS_DIR, OUTPUT_DIR, PROCESSED_DIR, OPENROUTER_API_KEY
from src.preprocessing.pdf_processor import PDFProcessor
from src.preprocessing.text_cleaner import TextCleaner
from src.llm_extraction.llm_client import LLMClient, MockLLMClient
from src.llm_extraction.entity_extractor import EntityExtractor
from src.graph_builder.neo4j_client import Neo4jClient
from src.graph_builder.graph_constructor import GraphConstructor
from src.visualization.graph_visualizer import GraphVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitedExtractor:
    """Wrapper for EntityExtractor with rate limit handling"""
    
    def __init__(self, extractor: EntityExtractor, delay_per_chunk: float = 3.0):
        self.extractor = extractor
        self.delay = delay_per_chunk
        self.processed_count = 0
    
    def extract_from_document_with_delay(self, document: dict, 
                                         max_chunks: int = None) -> dict:
        """
        Extract with delays between chunks
        
        Args:
            document: Preprocessed document
            max_chunks: Limit number of chunks (None = all)
        """
        logger.info(f"Processing: {document.get('filename', 'Unknown')}")
        
        chunks = document.get('chunks', [])
        if max_chunks:
            chunks = chunks[:max_chunks]
            logger.info(f"Limited to first {max_chunks} chunks")
        
        extractions = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Chunk {i}/{len(chunks)} - Waiting {self.delay}s before processing...")
            
            if i > 1:  # Don't wait before first chunk
                time.sleep(self.delay)
            
            try:
                extraction = self.extractor.extract_from_text(chunk)
                if extraction:
                    extractions.append(extraction)
                    logger.info(f"✓ Extracted {len(extraction['entities'])} entities, "
                              f"{len(extraction['relationships'])} relationships")
                    self.processed_count += 1
                else:
                    logger.warning(f"✗ No extraction for chunk {i}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
        
        # Merge extractions
        merged = self.extractor.merge_extractions(extractions)
        merged['document_name'] = document.get('filename', 'Unknown')
        merged['chunks_processed'] = len(extractions)
        
        return merged


def main():
    parser = argparse.ArgumentParser(description='Optimized Pipeline with Rate Limit Handling')
    
    parser.add_argument('--input', type=str, default=None, help='Input PDF directory')
    parser.add_argument('--delay', type=float, default=3.0, 
                       help='Delay between API calls (seconds)')
    parser.add_argument('--max-chunks', type=int, default=20,
                       help='Max chunks per document (None = all)')
    parser.add_argument('--use-mock', action='store_true', help='Use mock LLM')
    parser.add_argument('--resume-from', type=str, help='Resume from saved extraction file')
    
    args = parser.parse_args()
    
    print(f"\n{Fore.CYAN}=== Optimized Pipeline with Rate Limit Handling ==={Style.RESET_ALL}\n")
    
    # Initialize LLM
    if args.use_mock or not OPENROUTER_API_KEY:
        logger.info("Using Mock LLM (no API calls)")
        llm_client = MockLLMClient()
    else:
        logger.info(f"Using OpenRouter API with {args.delay}s delay between chunks")
        llm_client = LLMClient()
    
    extractor = EntityExtractor(llm_client)
    rate_limited_extractor = RateLimitedExtractor(extractor, delay_per_chunk=args.delay)
    
    # Check if resuming
    if args.resume_from:
        logger.info(f"Resuming from saved file: {args.resume_from}")
        with open(args.resume_from, 'r') as f:
            global_extraction = json.load(f)
        logger.info(f"Loaded: {len(global_extraction['entities'])} entities, "
                   f"{len(global_extraction['relationships'])} relationships")
    else:
        # Step 1: Load and preprocess
        print(f"\n{Fore.GREEN}[Step 1] Document Preprocessing{Style.RESET_ALL}")
        
        input_path = Path(args.input) if args.input else SAMPLE_REPORTS_DIR
        pdf_processor = PDFProcessor()
        documents = pdf_processor.process_directory(input_path)
        
        logger.info(f"Found {len(documents)} documents")
        
        preprocessed_docs = []
        for doc in documents:
            processed = TextCleaner.preprocess_document(doc, chunk_size=2000, overlap=200)
            preprocessed_docs.append(processed)
            logger.info(f"  {doc['filename']}: {processed['num_chunks']} chunks")
        
        # Step 2: Extract with rate limiting
        print(f"\n{Fore.GREEN}[Step 2] Rate-Limited Extraction{Style.RESET_ALL}")
        print(f"Delay: {args.delay}s per chunk | Max chunks: {args.max_chunks or 'all'}")
        
        all_extractions = []
        
        for i, doc in enumerate(preprocessed_docs, 1):
            print(f"\n--- Document {i}/{len(preprocessed_docs)} ---")
            
            extraction = rate_limited_extractor.extract_from_document_with_delay(
                doc, 
                max_chunks=args.max_chunks
            )
            all_extractions.append(extraction)
            
            # Save progress after each document
            progress_file = OUTPUT_DIR / f"extraction_progress_{i}.json"
            with open(progress_file, 'w') as f:
                json.dump(extraction, f, indent=2)
            logger.info(f"Progress saved to: {progress_file}")
        
        # Merge all extractions
        global_extraction = extractor.create_global_extraction(all_extractions)
        
        # Save final result
        output_file = OUTPUT_DIR / "extractions_complete.json"
        with open(output_file, 'w') as f:
            json.dump(global_extraction, f, indent=2)
        
        logger.info(f"\n✅ Extraction Complete!")
        logger.info(f"Total: {len(global_extraction['entities'])} entities, "
                   f"{len(global_extraction['relationships'])} relationships")
        logger.info(f"Saved to: {output_file}")
    
    # Step 3: Build graph
    print(f"\n{Fore.GREEN}[Step 3] Build Knowledge Graph{Style.RESET_ALL}")
    
    try:
        neo4j_client = Neo4jClient()
        neo4j_client.connect()
        
        constructor = GraphConstructor(neo4j_client)
        stats = constructor.build_graph_from_extraction(global_extraction, clear_existing=True)
        
        logger.info(f"Graph built: {stats['nodes_created']} nodes, "
                   f"{stats['relationships_created']} relationships")
        
        # Step 4: Visualize
        print(f"\n{Fore.GREEN}[Step 4] Create Visualizations{Style.RESET_ALL}")
        
        visualizer = GraphVisualizer(neo4j_client)
        viz_file = visualizer.create_pyvis_network("optimized_graph.html")
        
        logger.info(f"Visualization saved: {viz_file}")
        
        neo4j_client.close()
        
    except Exception as e:
        logger.error(f"Graph building failed: {e}")
        logger.info("You can still use the extracted data from JSON file")
    
    print(f"\n{Fore.GREEN}✅ Pipeline Complete!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()