"""
Lotto Data Scraper
Automatically updates Lotto.csv from the official archive
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime
import logging
import time
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../data/scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LottoScraper:
    def __init__(self):
        self.base_url = "https://www.pais.co.il/lotto/archive.aspx"
        self.csv_path = "../data/Lotto.csv"
        self.session = requests.Session()
        
        # Set proper headers to avoid being blocked
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def scrape_latest_results(self):
        """Scrape the latest lottery results from the official archive"""
        try:
            logger.info("Fetching latest lottery results")
            
            # Add delay to be respectful to the server
            time.sleep(1)
            
            # Get the main archive page
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug: Log the page structure
            logger.info("Successfully fetched page, parsing content...")
            
            # Find lottery result tables - updated selectors
            tables = soup.find_all('table', class_=re.compile(r'.*lotto.*', re.I))
            
            if not tables:
                # Try alternative selectors
                tables = soup.find_all('table')
                logger.info(f"Found {len(tables)} tables on page")
            
            results = []
            
            for i, table in enumerate(tables[:10]):  # Limit to first 10 tables
                try:
                    # Look for draw information in various possible formats
                    draw_info = self._extract_draw_info(table)
                    if draw_info:
                        results.append(draw_info)
                        logger.info(f"Extracted draw: {draw_info['draw_number']}")
                except Exception as e:
                    logger.warning(f"Error parsing table {i}: {e}")
                    continue
            
            if not results:
                logger.warning("No lottery results found on page")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            logger.info(f"Successfully scraped {len(df)} lottery draws")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Network error scraping results: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error scraping results: {e}")
            return pd.DataFrame()

    def _extract_draw_info(self, table):
        """Extract draw information from a table element"""
        try:
            # Look for draw number and date in various formats
            draw_number = None
            draw_date = None
            numbers = []
            
            # Try to find draw number and date
            for row in table.find_all('tr'):
                text = row.get_text(strip=True)
                
                # Look for draw number pattern
                draw_match = re.search(r'(?:הגרלה|גרלה|מס\')\s*(\d+)', text)
                if draw_match:
                    draw_number = draw_match.group(1)
                
                # Look for date pattern
                date_match = re.search(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text)
                if date_match:
                    draw_date = date_match.group(1)
                
                # Look for numbers
                number_cells = row.find_all(['td', 'th'])
                for cell in number_cells:
                    cell_text = cell.get_text(strip=True)
                    if cell_text.isdigit() and 1 <= int(cell_text) <= 45:
                        numbers.append(cell_text)
            
            # Validate we have enough information
            if not draw_number or len(numbers) < 6:
                return None
            
            # Format the result
            result = {
                'draw_number': draw_number,
                'draw_date': draw_date or 'Unknown',
                'number_1': numbers[0] if len(numbers) > 0 else '',
                'number_2': numbers[1] if len(numbers) > 1 else '',
                'number_3': numbers[2] if len(numbers) > 2 else '',
                'number_4': numbers[3] if len(numbers) > 3 else '',
                'number_5': numbers[4] if len(numbers) > 4 else '',
                'number_6': numbers[5] if len(numbers) > 5 else '',
                'strong_number': numbers[6] if len(numbers) > 6 else ''
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting draw info: {e}")
            return None

    def update_data_file(self):
        """Update the Lotto.csv file with new results"""
        try:
            # Scrape latest results
            new_data = self.scrape_latest_results()
            
            if new_data.empty:
                logger.warning("No new data scraped")
                return False
            
            # Load existing data if file exists
            if os.path.exists(self.csv_path):
                try:
                    existing_data = pd.read_csv(self.csv_path)
                    logger.info(f"Loaded {len(existing_data)} existing records")
                    
                    # Combine and remove duplicates based on draw_number
                    combined = pd.concat([new_data, existing_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['draw_number'], keep='first')
                    
                    # Sort by draw_number (descending for newest first)
                    combined['draw_number'] = pd.to_numeric(combined['draw_number'], errors='coerce')
                    combined = combined.sort_values('draw_number', ascending=False)
                    
                except Exception as e:
                    logger.error(f"Error loading existing data: {e}")
                    combined = new_data
            else:
                combined = new_data
                logger.info("No existing data file found, creating new one")
            
            # Create backup of existing file
            if os.path.exists(self.csv_path):
                backup_path = self.csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                os.rename(self.csv_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Save updated data
            combined.to_csv(self.csv_path, index=False)
            logger.info(f"Updated data file with {len(combined)} total draws")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating data file: {e}")
            return False

    def validate_data(self):
        """Validate the scraped data"""
        try:
            if not os.path.exists(self.csv_path):
                logger.error("Data file does not exist")
                return False
            
            df = pd.read_csv(self.csv_path)
            
            # Check for required columns
            required_cols = ['draw_number', 'draw_date', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for duplicate draw numbers
            duplicates = df[df.duplicated(subset=['draw_number'], keep=False)]
            if not duplicates.empty:
                logger.warning(f"Found {len(duplicates)} duplicate draw numbers")
            
            logger.info(f"Data validation completed. {len(df)} records found.")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

def main():
    """Main function to run the scraper"""
    scraper = LottoScraper()
    
    logger.info("Starting Lotto data scraper...")
    
    # Update data
    success = scraper.update_data_file()
    
    if success:
        # Validate the updated data
        scraper.validate_data()
        logger.info("Scraper completed successfully")
    else:
        logger.error("Scraper failed to update data")
    
    return success

if __name__ == "__main__":
    main()