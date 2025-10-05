"""
XLSX Handler: Extract text and data from Excel spreadsheets.
"""

from pathlib import Path
from typing import Dict, Any
import openpyxl
import pandas as pd


class XLSXHandler:
    """Handler for processing XLSX files."""
    
    def __init__(self):
        self.supported_extensions = ['.xlsx', '.xls']
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract data from an Excel spreadsheet.
        
        Args:
            file_path: Path to the XLSX file
            
        Returns:
            Dictionary containing text content and metadata
        """
        xlsx_path = Path(file_path)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        results = {
            'text_content': [],
            'images': [],
            'metadata': {
                'file_name': xlsx_path.name,
                'file_path': str(xlsx_path),
                'file_type': 'xlsx'
            }
        }
        
        try:
            # Load workbook
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            sheet_names = workbook.sheetnames
            
            results['metadata']['num_sheets'] = len(sheet_names)
            results['metadata']['sheet_names'] = sheet_names
            
            # Process each sheet
            for sheet_name in sheet_names:
                sheet_text = self._extract_sheet_data(file_path, sheet_name)
                if sheet_text:
                    results['text_content'].append({
                        'sheet': sheet_name,
                        'text': sheet_text
                    })
            
        except Exception as e:
            raise Exception(f"Error processing Excel file {file_path}: {str(e)}")
        
        return results
    
    def _extract_sheet_data(self, file_path: str, sheet_name: str) -> str:
        """
        Extract data from a specific sheet.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to extract
            
        Returns:
            Formatted text representation of the sheet data
        """
        try:
            # Read sheet with pandas for easier data handling
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Drop rows and columns that are completely empty
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                return ""
            
            # Convert to text representation
            # Include headers and format as readable text
            text_parts = [f"Sheet: {sheet_name}\n"]
            
            # Add column headers
            headers = ' | '.join([str(col) for col in df.columns])
            text_parts.append(headers)
            text_parts.append('-' * len(headers))
            
            # Add rows
            for _, row in df.iterrows():
                row_text = ' | '.join([str(val) if pd.notna(val) else '' for val in row])
                text_parts.append(row_text)
            
            return '\n'.join(text_parts)
            
        except Exception as e:
            print(f"Warning: Could not extract sheet {sheet_name}: {str(e)}")
            return ""
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file extension is supported."""
        return Path(file_path).suffix.lower() in self.supported_extensions
