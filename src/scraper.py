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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
    def __init__(self, data_dir=None):
        self.base_url = "https://www.pais.co.il/lotto/archive.aspx"
        if data_dir:
            self.csv_path = os.path.join(data_dir, "Lotto.csv")
            log_path = os.path.join(data_dir, "scraper.log")
        else:
            self.csv_path = "../data/Lotto.csv"
            log_path = "../data/scraper.log"
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Set up requests session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.3'
        })
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=chrome_options)
        logger.info("Selenium WebDriver initialized")
        
        # Set up logging
        logger = logging.getLogger(__name__)

    def __del__(self):
        """Ensure the WebDriver is properly closed"""
        try:
            self.driver.quit()
            logger.info("Selenium WebDriver closed")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")

    def scrape_latest_results(self):
        """Scrape the latest lottery results from the official archive using Selenium or direct download"""
        try:
            logger.info("Fetching latest lottery results")
            
            # First, try direct download of CSV if possible
            csv_url = "https://www.pais.co.il/lotto/archive.aspx#"  # Provided URL
            logger.info(f"Attempting direct download from {csv_url}")
            
            # Add delay to be respectful to the server
            time.sleep(1)
            
            # Try to download directly with requests
            try:
                response = self.session.get(csv_url, timeout=30)
                response.raise_for_status()
                
                # Check if response contains CSV data or needs further processing
                content_type = response.headers.get('content-type', '').lower()
                if 'csv' in content_type or 'text/csv' in content_type:
                    logger.info("Direct CSV download successful")
                    df = pd.read_csv(response.content)
                    logger.info(f"Successfully loaded {len(df)} lottery draws from CSV")
                    return df
                else:
                    logger.info("Direct download did not return CSV, proceeding with Selenium")
            except Exception as e:
                logger.warning(f"Direct download failed: {e}, falling back to Selenium")
            
            # If direct download fails, proceed with Selenium
            logger.info("Using Selenium for dynamic content")
            # Navigate to the page with Selenium
            self.driver.get(self.base_url)
            
            # Wait longer for dynamic content to load
            WebDriverWait(self.driver, 40).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            logger.info("Page loaded with Selenium")
            
            # Try to interact with the page to load results (e.g., click a button or select an option)
            try:
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                for btn in buttons:
                    btn_text = btn.text.strip()
                    if btn_text and ("תוצאות" in btn_text or "results" in btn_text.lower() or "הגרלה" in btn_text):
                        btn.click()
                        logger.info(f"Clicked button with text: {btn_text}")
                        time.sleep(2)  # Wait for content to load after click
            except Exception as e:
                logger.warning(f"Error interacting with buttons: {e}")
            
            # Parse the rendered HTML content
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Log page title for context
            title = soup.title.get_text(strip=True) if soup.title else "No title found"
            logger.info(f"Page title: {title}")
            
            # Find lottery result elements - try various structures
            result_elements = soup.find_all(['div', 'li', 'span', 'table'], class_=re.compile(r'.*lotto.*|.*result.*|.*draw.*|.*number.*', re.I))
            logger.info(f"Found {len(result_elements)} potential result elements")
            
            if not result_elements:
                # Log all divs with potential content for debugging
                divs = soup.find_all('div', class_=True)
                div_classes = [div.get('class', []) for div in divs[:10]]  # Limit to first 10
                logger.info(f"Sample div classes on page: {div_classes}")
                logger.warning("No lottery results found on page")
                body_text = soup.body.get_text(strip=True)[:500] if soup.body else "No body content"
                logger.debug(f"Page content sample: {body_text}")
                return pd.DataFrame()
            
            results = self._extract_results_from_elements(result_elements)
            if not results:
                logger.warning("No valid lottery results extracted")
                # Log content of some elements for debugging
                for i, elem in enumerate(result_elements[:10]):  # Limit to first 10
                    elem_text = elem.get_text(strip=True)[:100]  # Limit text length
                    logger.debug(f"Element {i} content: {elem_text}")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            logger.info(f"Successfully scraped {len(df)} lottery draws")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping results with Selenium: {e}")
            return pd.DataFrame()

    def _extract_results_from_elements(self, elements):
        """Extract lottery results from non-table elements like divs or spans"""
        try:
            results = []
            current_draw = None
            numbers = []
            strong_number = None
            draw_number = None
            draw_date = None
            
            for elem in elements:
                text = elem.get_text(strip=True)
                if not text:
                    continue
                
                logger.debug(f"Processing element text: {text}")
                
                # Look for draw number
                if 'הגרלה' in text or 'Draw' in text or 'מספר' in text:
                    match = re.search(r'\d{3,5}', text)  # Broader range for draw numbers
                    if match:
                        if current_draw and len(numbers) >= 6:
                            # Save previous draw if exists
                            results.append({
                                'draw_number': current_draw,
                                'draw_date': draw_date or '',
                                'number_1': numbers[0] if len(numbers) > 0 else '',
                                'number_2': numbers[1] if len(numbers) > 1 else '',
                                'number_3': numbers[2] if len(numbers) > 2 else '',
                                'number_4': numbers[3] if len(numbers) > 3 else '',
                                'number_5': numbers[4] if len(numbers) > 4 else '',
                                'number_6': numbers[5] if len(numbers) > 5 else '',
                                'strong_number': strong_number if strong_number else ''
                            })
                            logger.info(f"Saved draw {current_draw} with {len(numbers)} numbers")
                        draw_number = match.group()
                        current_draw = draw_number
                        numbers = []
                        strong_number = None
                        draw_date = None
                
                # Look for date
                if not draw_date:
                    date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
                    if date_match:
                        draw_date = date_match.group()
                        parts = re.split(r'[-/]', draw_date)
                        if len(parts[-1]) == 2:
                            parts[-1] = '20' + parts[-1]
                        draw_date = '/'.join(parts)
                
                # Look for numbers - be more aggressive
                num_match = re.findall(r'\b\d{1,2}\b', text)
                for num_text in num_match:
                    num = int(num_text)
                    if 1 <= num <= 37 and num not in numbers:  # Avoid duplicates
                        numbers.append(num)
                        logger.debug(f"Added number {num} to draw {current_draw}")
                
                # Look for strong number
                if 'חזק' in text or 'strong' in text.lower():
                    strong_match = re.search(r'\d+', text)
                    if strong_match:
                        strong_number = strong_match.group()
                        logger.debug(f"Found strong number {strong_number} for draw {current_draw}")
                    else:
                        next_elem = elem.find_next()
                        if next_elem:
                            next_text = next_elem.get_text(strip=True)
                            strong_match = re.search(r'\d+', next_text)
                            if strong_match:
                                strong_number = strong_match.group()
                                logger.debug(f"Found strong number {strong_number} in next element for draw {current_draw}")
            
            # Don't forget to add the last draw if exists
            if current_draw and len(numbers) >= 6:
                results.append({
                    'draw_number': current_draw,
                    'draw_date': draw_date or '',
                    'number_1': numbers[0] if len(numbers) > 0 else '',
                    'number_2': numbers[1] if len(numbers) > 1 else '',
                    'number_3': numbers[2] if len(numbers) > 2 else '',
                    'number_4': numbers[3] if len(numbers) > 3 else '',
                    'number_5': numbers[4] if len(numbers) > 4 else '',
                    'number_6': numbers[5] if len(numbers) > 5 else '',
                    'strong_number': strong_number if strong_number else ''
                })
                logger.info(f"Saved last draw {current_draw} with {len(numbers)} numbers")
            
            # Filter out incomplete results
            valid_results = [r for r in results if r['number_1'] and r['number_2'] and r['number_3'] and r['number_4'] and r['number_5'] and r['number_6']]
            logger.info(f"Extracted {len(valid_results)} valid draws from elements")
            return valid_results
        except Exception as e:
            logger.error(f"Error extracting results from elements: {e}")
            return []

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