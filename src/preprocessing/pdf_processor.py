"""
PDF Processing Module
Handles extraction of text from PDF documents
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract and process text from PDF files"""
    
    def __init__(self, method='pymupdf'):
        """
        Initialize PDF processor
        
        Args:
            method: 'pymupdf' or 'pdfplumber'
        """
        self.method = method
    
    def extract_text_pymupdf(self, pdf_path: Path) -> str:
        """
        Extract text using PyMuPDF (faster, good for most PDFs)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """
        Extract text using pdfplumber (better for complex layouts)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using selected method
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if self.method == 'pymupdf':
            return self.extract_text_pymupdf(pdf_path)
        else:
            return self.extract_text_pdfplumber(pdf_path)
    
    def extract_metadata(self, pdf_path: Path) -> Dict:
        """
        Extract metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            metadata['num_pages'] = len(doc)
            doc.close()
            return metadata
        
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {}
    
    def process_directory(self, directory: Path) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Path to directory containing PDFs
            
        Returns:
            List of dictionaries with filename, text, and metadata
        """
        results = []
        pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path.name}")
            
            text = self.extract_text(pdf_path)
            metadata = self.extract_metadata(pdf_path)
            
            results.append({
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'text': text,
                'metadata': metadata,
                'text_length': len(text)
            })
        
        return results


def create_sample_text_documents():
    """
    Create sample financial report texts for demonstration
    
    Returns:
        List of sample document dictionaries
    """
    sample_docs = [
        {
            'filename': 'samsung_2024_report.txt',
            'text': """
Samsung Electronics Q4 2024 Financial Report

Samsung Electronics, headquartered in Seoul, South Korea, reported record revenues 
of $245 billion for fiscal year 2024. The company's CEO, Jong-Hee Han, announced 
strategic investments in AI chip technology and quantum computing.

Key Developments:
- Samsung invested $2.5 billion in AI startup Cerebras Systems
- Acquired Dutch semiconductor equipment manufacturer ASML's subsidiary for $1.2B
- Launched Galaxy AI platform powered by proprietary Exynos 3000 processor
- Partnered with Microsoft for cloud computing solutions in Asian markets

Subsidiary Performance:
Samsung Display reported revenue of $28 billion, primarily from OLED panel sales 
to Apple and other smartphone manufacturers. Samsung Biologics, the pharmaceutical 
subsidiary, secured a $500 million contract with Pfizer for antibody production.

Financial Metrics:
- Operating profit: $42 billion (up 15% YoY)
- R&D spending: $22 billion (9% of revenue)
- Employee count: 267,000 globally
- Market cap: $380 billion as of December 2024

Samsung competes directly with TSMC in advanced chip manufacturing and maintains 
partnerships with Google for Android optimization and Qualcomm for 5G technology.
            """,
            'metadata': {'author': 'Samsung IR', 'year': 2024}
        },
        {
            'filename': 'apple_innovation_report.txt',
            'text': """
Apple Inc. Innovation and Investment Report 2024

Apple Inc., led by CEO Tim Cook, continues to dominate consumer technology markets
with headquarters in Cupertino, California. The company manages a diverse portfolio
of products and services generating $394 billion in annual revenue.

Strategic Investments:
- Apple invested $500 million in renewable energy projects in partnership with 
  Tesla Energy Solutions
- Acquired autonomous vehicle startup Drive.ai for $200 million
- Established Apple Semiconductor Research Center in Austin, Texas
- Invested $1 billion in Indian manufacturing expansion

Product Ecosystem:
Apple's flagship iPhone 16 uses the A18 Bionic chip manufactured by TSMC. The 
company's services division, including Apple Music, iCloud, and Apple TV+, 
generated $85 billion in revenue.

Key Partnerships:
- Long-term supply agreement with Samsung Display for OLED screens
- Collaboration with OpenAI to integrate GPT models into Siri
- Partnership with Goldman Sachs for Apple Card and savings accounts
- Joint venture with Hyundai for Apple Car development

Executive Team:
Tim Cook (CEO), Luca Maestri (CFO), and Jeff Williams (COO) lead the company.
Greg Joswiak manages worldwide marketing, while John Ternus oversees hardware
engineering including the Vision Pro augmented reality headset.

Apple competes with Samsung in smartphones, Microsoft in tablets and services,
and Google in digital services and smart home technology.
            """,
            'metadata': {'author': 'Apple Investor Relations', 'year': 2024}
        },
        {
            'filename': 'tesla_expansion_summary.txt',
            'text': """
Tesla Corporation Expansion Summary 2024

Tesla, under CEO Elon Musk, expanded operations across multiple continents with
headquarters remaining in Austin, Texas. The electric vehicle and energy company
achieved $96 billion in annual revenue.

Manufacturing Footprint:
- Gigafactory Berlin produced 500,000 vehicles in 2024
- Gigafactory Shanghai: 750,000 vehicles (partnership with CATL for batteries)
- Gigafactory Texas: 400,000 Cybertrucks and Model Y vehicles
- New facility in Monterrey, Mexico under construction

Technology Development:
Tesla's Full Self-Driving (FSD) system uses custom AI chips designed in partnership
with Nvidia. The company acquired computer vision startup DeepScale for $150 million
to enhance autonomous driving capabilities.

Energy Division:
Tesla Energy, managed by Drew Baglino, deployed 15 GWh of energy storage products
globally. Major projects include:
- 500 MW Megapack installation for California utilities (PG&E partnership)
- Solar roof installations in partnership with SolarCity (Tesla subsidiary)
- Powerwall home battery systems sold through partnerships with Home Depot

Investment Activity:
- Tesla invested $800 million in lithium mining operations in Nevada
- Acquired battery recycling startup Redwood Materials for $500 million
- Partnership with Panasonic for 4680 battery cell production
- Joint venture with BYD for affordable EV models in emerging markets

Competition and Partnerships:
Tesla competes with traditional automakers like Ford, GM, and Volkswagen while
partnering with Uber for autonomous taxi services. The company supplies battery
technology to Mercedes-Benz and BMW through licensing agreements.

Financial Performance:
- Net income: $12 billion
- R&D expenditure: $3.5 billion
- Production capacity: 2 million vehicles annually
- Supercharger network: 50,000 stations worldwide
            """,
            'metadata': {'author': 'Tesla Communications', 'year': 2024}
        }
    ]
    
    return sample_docs