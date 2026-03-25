"""
PDF Parser Module - Extracts traffic data from PDF reports
"""

import pdfplumber
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PDFParser:
    """Parse traffic dataset PDFs"""
    
    def __init__(self):
        self.data = {
            'metadata': {},
            'violations': [],
            'traffic_volume': [],
            'intersection_info': {},
            'pedestrian_data': []
        }
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF file and extract structured data
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Structured data dictionary
        """
        logger.info(f"Parsing PDF: {file_path}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                self.data['metadata'] = {
                    'pages': len(pdf.pages),
                    'file_name': file_path.split('/')[-1],
                    'parsed_at': datetime.now().isoformat()
                }
                
                full_text = ""
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    full_text += text
                    
                    tables = page.extract_tables()
                    for table in tables:
                        self._process_table(table)
                
                self._extract_text_info(full_text)
                
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise
        
        return self.data
    
    def _process_table(self, table: List[List]) -> None:
        """Process a table from PDF"""
        if not table or len(table) < 2:
            return
        
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
        
        for row in table[1:]:
            if not any(row):
                continue
            
            row_dict = {}
            for i, cell in enumerate(row):
                if i < len(headers) and cell:
                    val = str(cell).strip()
                    if val:
                        row_dict[headers[i]] = val
            
            if row_dict:
                self._categorize_row(row_dict)
    
    def _categorize_row(self, row: Dict) -> None:
        """Categorize row into appropriate data type"""
        row_text = str(row).lower()
        
        if any(k in row_text for k in ['violation', 'block', 'vehicle_id', 'id', 'duration']):
            self.data['violations'].append(row)
        elif any(k in row_text for k in ['volume', 'count', 'hourly', 'traffic']):
            self.data['traffic_volume'].append(row)
        elif any(k in row_text for k in ['pedestrian', 'crossing', 'ped']):
            self.data['pedestrian_data'].append(row)
    
    def _extract_text_info(self, text: str) -> None:
        """Extract information from text"""
        text_lower = text.lower()
        
        # Extract intersection name
        patterns = [
            r'intersection[:\s]+([A-Za-z\s&]+)',
            r'location[:\s]+([A-Za-z\s&]+)',
            r'junction[:\s]+([A-Za-z\s&]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                self.data['intersection_info']['name'] = match.group(1).strip()
                break
        
        # Default if not found
        if not self.data['intersection_info'].get('name'):
            self.data['intersection_info']['name'] = "Banashankari Junction"
        
        # Extract violation counts
        count_patterns = [
            r'(\d+)\s*violations?',
            r'(\d+)\s*vehicles?\s*blocking',
            r'total.*?(\d+)\s*incidents'
        ]
        
        for pattern in count_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    self.data['metadata']['violation_count'] = int(match.group(1))
                    break
                except ValueError:
                    pass
        
        # Extract dates
        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
        dates = re.findall(date_pattern, text)
        if dates:
            self.data['metadata']['dates'] = dates[:5]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of parsed data"""
        return {
            'total_violations': len(self.data['violations']),
            'total_volume_records': len(self.data['traffic_volume']),
            'intersection': self.data['intersection_info'].get('name', 'Unknown'),
            'dates': self.data['metadata'].get('dates', [])
        }