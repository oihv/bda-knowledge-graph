#!/usr/bin/env python3
"""
Test script for rate limit optimizations
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.text_cleaner import TextCleaner

def test_chunking_optimizations():
    """Test the chunking and rate limiting optimizations"""
    
    # Create a large sample text (simulating Apple 10-Q size)
    sample_text = """
    Apple Inc. Financial Report Q4 2024
    
    Executive Summary:
    Apple Inc. reported record quarterly revenue of $119.6 billion for Q4 2024, 
    representing 7% year-over-year growth. The company's CEO Tim Cook announced 
    strategic investments in AI and autonomous systems technology.
    
    Key Developments:
    Apple acquired AI startup Turi for $200 million to enhance machine learning 
    capabilities. The company also partnered with Tesla for autonomous vehicle 
    technology development. Major product launches included iPhone 16 Pro and 
    Apple Vision Pro 2.
    
    Financial Performance:
    Revenue segments showed strong performance across all categories:
    - iPhone: $46.2 billion (up 6%)
    - Services: $22.3 billion (up 12%) 
    - Mac: $7.1 billion (up 2%)
    - iPad: $6.4 billion (down 7%)
    - Wearables: $9.3 billion (up 3%)
    
    The company's gross margin increased to 46.2%, driven by higher Services revenue 
    and improved supply chain efficiency. Operating expenses were $13.4 billion.
    
    Strategic Investments:
    Apple invested $500 million in renewable energy projects through partnerships 
    with SolarCity and Tesla Energy. The company also established new R&D centers 
    in Austin, Texas and Bangalore, India.
    
    Partnerships and Acquisitions:
    Key partnerships were formed with Samsung Display for OLED technology, 
    Qualcomm for 5G modems, and OpenAI for Siri improvements. Apple acquired 
    autonomous driving startup Drive.ai for $200 million.
    """ * 50  # Multiply to simulate large document
    
    print(f"Sample text length: {len(sample_text):,} characters")
    
    # Test 1: Unlimited chunking (old behavior)
    print("\n=== Test 1: Unlimited Chunking ===")
    chunks_unlimited = TextCleaner.chunk_text(sample_text, chunk_size=2000, overlap=200)
    print(f"Unlimited chunks: {len(chunks_unlimited)}")
    print(f"Estimated API calls: {len(chunks_unlimited)}")
    print(f"Estimated tokens: {len(chunks_unlimited) * 3500:,}")
    
    # Test 2: Limited chunking (new behavior)
    print("\n=== Test 2: Limited Chunking (Rate Limit Friendly) ===")
    chunks_limited = TextCleaner.chunk_text(sample_text, chunk_size=2000, overlap=200, max_chunks=20)
    print(f"Limited chunks: {len(chunks_limited)}")
    print(f"Estimated API calls: {len(chunks_limited)}")
    print(f"Estimated tokens: {len(chunks_limited) * 3500:,}")
    
    # Test 3: Document preprocessing with limits
    print("\n=== Test 3: Document Preprocessing with Smart Sampling ===")
    document = {
        'filename': 'test_apple_10q.pdf',
        'text': sample_text,
        'metadata': {}
    }
    
    processed = TextCleaner.preprocess_document(document, max_chunks=20)
    print(f"Processed chunks: {processed['num_chunks']}")
    print(f"Sections found: {len(processed['sections'])}")
    print(f"Final text length: {len(processed['cleaned_text']):,} characters")
    
    print(f"\n✅ Rate limiting optimizations implemented!")
    print(f"   Chunk reduction: {len(chunks_unlimited)} → {len(chunks_limited)} ({100*(1-len(chunks_limited)/len(chunks_unlimited)):.1f}% reduction)")
    print(f"   Token reduction: ~{len(chunks_unlimited)*3500:,} → ~{len(chunks_limited)*3500:,}")

if __name__ == "__main__":
    test_chunking_optimizations()