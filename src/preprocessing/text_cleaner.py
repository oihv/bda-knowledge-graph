"""
Text Cleaning and Chunking Module
Prepares extracted text for LLM processing
"""
import re
from typing import List, Dict, Optional
import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP

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
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, max_chunks: int = 0) -> List[str]:
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
    
    # NEW METHOD: Detect sections with more structure
    @staticmethod
    def detect_sections(text: str) -> List[Dict]:
        """
        Detect document sections based on formatting (headings, paragraphs)
        Returns structured sections with heading and content
        
        Args:
            text: Document text
            
        Returns:
            List of section dictionaries with 'heading' and 'content' keys
        """
        sections = []
        lines = text.split('\n')
        current_section = {'heading': '', 'content': [], 'section_type': 'body'}
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Detect headings - multiple heuristics
            is_heading = False
            
            # 1. All caps line (common in reports)
            if line_stripped.isupper() and len(line_stripped.split()) < 10 and len(line_stripped) > 3:
                is_heading = True
            
            # 2. Short line ending with colon
            elif line_stripped.endswith(':') and len(line_stripped.split()) < 8:
                is_heading = True
            
            # 3. Known section keywords
            elif any(keyword in line_stripped.lower() for keyword in 
                    ['executive summary', 'financial highlights', 'key metrics', 
                     'business overview', 'strategy', 'outlook', 'risk factors']):
                is_heading = True
            
            if is_heading:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'heading': line_stripped,
                    'content': [],
                    'section_type': 'heading'
                }
            else:
                # Add to current section content
                current_section['content'].append(line_stripped)
        
        # Add final section
        if current_section['content']:
            sections.append(current_section)
        
        # If no sections detected, create artificial sections based on length
        if not sections:
            # Split into manageable sections of roughly CHUNK_SIZE length
            text_lines = text.split('\n')
            current_section = {
                'heading': 'Section 1',
                'content': [],
                'section_type': 'body'
            }
            current_length = 0
            section_num = 1
            
            for line in text_lines:
                line_length = len(line)
                if current_length + line_length > CHUNK_SIZE:
                    # Save current section
                    if current_section['content']:
                        sections.append(current_section)
                    # Start new section
                    section_num += 1
                    current_section = {
                        'heading': f'Section {section_num}',
                        'content': [line],
                        'section_type': 'body'
                    }
                    current_length = line_length
                else:
                    current_section['content'].append(line)
                    current_length += line_length
            
            # Add final section
            if current_section['content']:
                sections.append(current_section)
        
        return sections
    
    # NEW METHOD: Create semantic chunks that respect section boundaries
    @staticmethod
    def create_semantic_chunks(sections: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
        """
        Create chunks that don't break sections when possible
        Returns chunks with metadata
        
        Args:
            sections: List of section dictionaries from detect_sections()
            chunk_size: Target characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        current_chunk = {
            'text': '',
            'sections': [],
            'has_heading': False
        }
        
        for section in sections:
            # Reconstruct section text
            section_text = ''
            if section['heading']:
                section_text = f"{section['heading']}\n"
                current_chunk['has_heading'] = True
            section_text += '\n'.join(section['content'])
            
            section_length = len(section_text)
            current_length = len(current_chunk['text'])
            
            # If adding this section would exceed chunk_size
            if current_length + section_length > chunk_size and current_length > 0:
                # Save current chunk
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk['text']:
                    overlap_text = current_chunk['text'][-overlap:]
                    current_chunk = {
                        'text': overlap_text + '\n\n' + section_text,
                        'sections': [section['heading']],
                        'has_heading': bool(section['heading'])
                    }
                else:
                    current_chunk = {
                        'text': section_text,
                        'sections': [section['heading']],
                        'has_heading': bool(section['heading'])
                    }
            else:
                # Add to current chunk
                if current_chunk['text']:
                    current_chunk['text'] += '\n\n' + section_text
                else:
                    current_chunk['text'] = section_text
                
                if section['heading']:
                    current_chunk['sections'].append(section['heading'])
        
        # Add final chunk
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        return chunks
    
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
    def preprocess_document(document: Dict, chunk_size: int = CHUNK_SIZE, 
                          overlap: int = CHUNK_OVERLAP, max_chunks: int = 0) -> Dict:
        """
        Full preprocessing pipeline for a document
        NOW INCLUDES: Section-based chunking + document metadata in chunks
        
        Args:
            document: Document dictionary with 'text' key
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            max_chunks: Maximum number of chunks (0 for unlimited)
            
        Returns:
            Processed document with cleaned text and chunks WITH METADATA
        """
        logger.info(f"Preprocessing document: {document.get('filename', 'Unknown')}")
        
        # Clean text
        cleaned_text = TextCleaner.clean_text(document['text'])
        
        # Extract sections (old method for compatibility)
        sections = TextCleaner.extract_sections(cleaned_text)
        
        # NEW: Detect structured sections
        structured_sections = TextCleaner.detect_sections(cleaned_text)
        
        # Try semantic chunking first
        semantic_chunks = TextCleaner.create_semantic_chunks(
            structured_sections, 
            chunk_size, 
            overlap
        )
        
        # If semantic chunking produced too few chunks relative to text size,
        # fall back to basic chunking
        expected_chunks = len(cleaned_text) // chunk_size + (1 if len(cleaned_text) % chunk_size else 0)
        if len(semantic_chunks) < expected_chunks // 2:  # If we got less than half the expected chunks
            logger.info(f"Semantic chunking produced too few chunks ({len(semantic_chunks)} vs expected {expected_chunks})")
            logger.info("Falling back to basic chunking")
            basic_chunks = TextCleaner.chunk_text(cleaned_text, chunk_size, overlap)
            semantic_chunks = [{'text': chunk, 'sections': ['Basic Chunk'], 'has_heading': False} 
                             for chunk in basic_chunks]
        
        # Limit chunks if specified
        if max_chunks > 0 and len(semantic_chunks) > max_chunks:
            logger.warning(f"Limiting chunks from {len(semantic_chunks)} to {max_chunks}")
            semantic_chunks = semantic_chunks[:max_chunks]
        
        # NEW: Add document metadata to each chunk
        enhanced_chunks = []
        for idx, chunk_data in enumerate(semantic_chunks):
            enhanced_chunk = {
                'text': chunk_data['text'],
                'chunk_index': idx,
                'total_chunks': len(semantic_chunks),
                'document_name': document.get('filename', 'unknown'),
                'document_type': document.get('metadata', {}).get('type', 'financial_report'),
                'sections_included': chunk_data.get('sections', []),
                'has_heading': chunk_data.get('has_heading', False)
            }
            enhanced_chunks.append(enhanced_chunk)
        
        # Add processed data to document
        processed_doc = document.copy()
        processed_doc['cleaned_text'] = cleaned_text
        processed_doc['sections'] = sections
        processed_doc['structured_sections'] = structured_sections  # NEW
        processed_doc['chunks'] = enhanced_chunks  # NOW WITH METADATA
        processed_doc['num_chunks'] = len(enhanced_chunks)
        
        logger.info(f"Created {len(enhanced_chunks)} semantic chunks from {len(cleaned_text)} characters")
        
        return processed_doc