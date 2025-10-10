"""
Text Cleaning and Chunking Module
Prepares extracted text for LLM processing
"""
import re
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Clean and prepare text for processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text)
        
        # Remove headers/footers that repeat
        lines = text.split('\n')
        if len(lines) > 10:
            # Simple heuristic: if a line appears more than 3 times, it's likely a header/footer
            line_counts = {}
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) > 10:
                    line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
            
            filtered_lines = [
                line for line in lines 
                if line.strip() not in line_counts or line_counts[line.strip()] <= 3
            ]
            text = '\n'.join(filtered_lines)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\$\%\(\)\&]', '', text)
        
        # Normalize currency symbols
        text = re.sub(r'\$\s+', '$', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod  
    def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200, max_chunks: int = 0) -> List[str]:
        """
        Split text into overlapping chunks for LLM processing
        
        Args:
            text: Cleaned text
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            max_chunks: Maximum number of chunks to create (0 for unlimited)
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start, end - 200)
                sentence_end = max(
                    text.rfind('. ', search_start, end),
                    text.rfind('! ', search_start, end),
                    text.rfind('? ', search_start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end
            
            # Stop if we've reached the maximum number of chunks
            if max_chunks > 0 and len(chunks) >= max_chunks:
                logger.warning(f"Reached maximum chunk limit of {max_chunks}. Truncating document.")
                break
        
        return chunks
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Attempt to extract logical sections from text
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Common section headers in financial reports
        section_patterns = [
            r'(?:^|\n)(Executive Summary|Summary)[\:\s]',
            r'(?:^|\n)(Financial\s+(?:Performance|Results|Highlights))[\:\s]',
            r'(?:^|\n)(Key\s+(?:Developments|Highlights|Metrics))[\:\s]',
            r'(?:^|\n)(Investment(?:s)?|Strategic\s+Investment(?:s)?)[\:\s]',
            r'(?:^|\n)(Partnership(?:s)?|Collaboration(?:s)?)[\:\s]',
            r'(?:^|\n)(Acquisition(?:s)?|Merger(?:s)?)[\:\s]',
            r'(?:^|\n)(Subsidiary|Subsidiaries)[\:\s]',
        ]
        
        # Find all section starts
        section_markers = []
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_markers.append((match.start(), match.group(1)))
        
        # Sort by position
        section_markers.sort()
        
        # Extract content between markers
        for i, (start, name) in enumerate(section_markers):
            next_start = section_markers[i + 1][0] if i + 1 < len(section_markers) else len(text)
            content = text[start:next_start].strip()
            sections[name.strip()] = content
        
        # If no sections found, use entire text
        if not sections:
            sections['Full Document'] = text
        
        return sections
    
    @staticmethod
    def prioritize_sections(sections: Dict[str, str], max_chars: int = 100000) -> str:
        """
        Prioritize and sample important sections to reduce token usage
        
        Args:
            sections: Dictionary of section name to content
            max_chars: Maximum characters to return
            
        Returns:
            Concatenated priority sections
        """
        # Priority order for financial documents
        priority_sections = [
            'Executive Summary', 'Summary', 'Financial Performance', 
            'Financial Results', 'Financial Highlights', 'Key Developments',
            'Key Highlights', 'Key Metrics', 'Investment', 'Investments',
            'Strategic Investment', 'Strategic Investments', 'Partnership',
            'Partnerships', 'Collaboration', 'Collaborations', 'Acquisition',
            'Acquisitions', 'Merger', 'Mergers'
        ]
        
        result = ""
        remaining_chars = max_chars
        
        # First, add priority sections
        for priority in priority_sections:
            for section_name, content in sections.items():
                if priority.lower() in section_name.lower() and len(content) <= remaining_chars:
                    result += f"\n\n=== {section_name} ===\n{content}"
                    remaining_chars -= len(content) + 20  # Account for headers
                    break
        
        # Then add other sections if space remains
        for section_name, content in sections.items():
            if remaining_chars <= 0:
                break
            if not any(priority.lower() in section_name.lower() for priority in priority_sections):
                content_to_add = content[:remaining_chars] if len(content) > remaining_chars else content
                result += f"\n\n=== {section_name} ===\n{content_to_add}"
                remaining_chars -= len(content_to_add) + 20
        
        return result.strip()
    
    @staticmethod
    def preprocess_document(document: Dict, chunk_size: int = 2000, 
                          overlap: int = 200, max_chunks: int = 0) -> Dict:
        """
        Full preprocessing pipeline for a document
        
        Args:
            document: Document dictionary with 'text' key
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            max_chunks: Maximum number of chunks (0 for unlimited)
            
        Returns:
            Processed document with cleaned text and chunks
        """
        logger.info(f"Preprocessing document: {document.get('filename', 'Unknown')}")
        
        # Clean text
        cleaned_text = TextCleaner.clean_text(document['text'])
        
        # Extract sections
        sections = TextCleaner.extract_sections(cleaned_text)
        
        # Prioritize important sections if max_chunks is specified
        if max_chunks > 0:
            max_chars = max_chunks * chunk_size  # Estimate based on chunk limit
            prioritized_text = TextCleaner.prioritize_sections(sections, max_chars)
            if prioritized_text:
                cleaned_text = prioritized_text
                logger.info(f"Prioritized text reduced to {len(cleaned_text)} characters")
        
        # Chunk text
        chunks = TextCleaner.chunk_text(cleaned_text, chunk_size, overlap, max_chunks)
        
        # Add processed data to document
        processed_doc = document.copy()
        processed_doc['cleaned_text'] = cleaned_text
        processed_doc['sections'] = sections
        processed_doc['chunks'] = chunks
        processed_doc['num_chunks'] = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(cleaned_text)} characters")
        
        return processed_doc