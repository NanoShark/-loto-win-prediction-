import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LottoScraper:
    def __init__(self, data_dir='../data'):
        self.base_url = "https://www.pais.co.il/lotto/archive.aspx"
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, 'lotto_results.csv')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
    def scrape_results(self, max_pages=100):
        """
        Scrape lottery results from the Pais website
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            DataFrame containing the lottery results
        """
        logger.info("Starting to scrape lottery results")
        
        all_results = []
        page = 1
        
        while page <= max_pages:
            try:
                logger.info(f"Scraping page {page}")
                
                # Make request to the website
                if page == 1:
                    url = self.base_url
                else:
                    url = f"{self.base_url}?page={page}"
                
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find the table containing lottery results
                table = soup.find('table', class_='archive_table')
                
                if not table:
                    logger.info(f"No more results found on page {page}. Stopping.")
                    break
                
                # Extract rows from the table
                rows = table.find_all('tr')[1:]  # Skip header row
                
                if not rows:
                    logger.info(f"No rows found on page {page}. Stopping.")
                    break
                
                # Process each row
                for row in rows:
                    cells = row.find_all('td')
                    
                    if len(cells) < 9:  # Ensure we have enough cells
                        continue
                    
                    # Extract data from cells
                    try:
                        draw_date = cells[0].text.strip()
                        draw_number = cells[1].text.strip()
                        
                        # Extract the 6 regular numbers and the strong number
                        regular_numbers = []
                        for i in range(2, 8):
                            number = cells[i].text.strip()
                            regular_numbers.append(int(number))
                        
                        strong_number = int(cells[8].text.strip())
                        
                        # Add to results
                        result = {
                            'draw_date': draw_date,
                            'draw_number': draw_number,
                            'number_1': regular_numbers[0],
                            'number_2': regular_numbers[1],
                            'number_3': regular_numbers[2],
                            'number_4': regular_numbers[3],
                            'number_5': regular_numbers[4],
                            'number_6': regular_numbers[5],
                            'strong_number': strong_number
                        }
                        
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                
                # Move to next page
                page += 1
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Convert date strings to datetime objects
        df['draw_date'] = pd.to_datetime(df['draw_date'], format='%d/%m/%Y', errors='coerce')
        
        # Sort by draw date (newest first)
        df = df.sort_values('draw_date', ascending=False)
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        
        logger.info(f"Scraped {len(df)} lottery results and saved to {self.csv_path}")
        
        return df
    
    def load_results(self):
        """
        Load lottery results from CSV file if it exists, otherwise scrape them
        
        Returns:
            DataFrame containing the lottery results
        """
        if os.path.exists(self.csv_path):
            logger.info(f"Loading lottery results from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            # Convert date strings to datetime objects
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            
            return df
        else:
            logger.info(f"No existing results found at {self.csv_path}. Scraping new data.")
            return self.scrape_results()
    
    def update_results(self, force_update=False):
        """
        Update lottery results by scraping the latest data
        
        Args:
            force_update: If True, always scrape new data and delete old CSV
            
        Returns:
            DataFrame containing the updated lottery results
        """
        # If force update is requested, delete the existing CSV file
        if force_update and os.path.exists(self.csv_path):
            try:
                logger.info(f"Deleting existing CSV file at {self.csv_path}")
                os.remove(self.csv_path)
                logger.info("Existing CSV file deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting CSV file: {e}")
        
        # If force update or no existing data, scrape all results
        if force_update or not os.path.exists(self.csv_path):
            logger.info("Scraping all lottery results from the Pais website")
            df = self.scrape_results()
            logger.info(f"Scraped {len(df)} lottery results")
            return df
        
        # Otherwise, check if we need to update based on the latest draw date
        try:
            # Try to read with iso-8859-1 encoding first
            existing_df = pd.read_csv(self.csv_path, encoding='iso-8859-1')
            
            # Rename columns if needed
            if 'draw_date' not in existing_df.columns:
                # The first column is the draw number, second is the date, then 6 regular numbers, then the strong number
                column_mapping = {
                    existing_df.columns[0]: 'draw_number',
                    existing_df.columns[1]: 'draw_date',
                    existing_df.columns[2]: 'number_1',
                    existing_df.columns[3]: 'number_2',
                    existing_df.columns[4]: 'number_3',
                    existing_df.columns[5]: 'number_4',
                    existing_df.columns[6]: 'number_5',
                    existing_df.columns[7]: 'number_6',
                    existing_df.columns[8]: 'strong_number'
                }
                existing_df = existing_df.rename(columns=column_mapping)
            
            # Convert date strings to datetime objects
            existing_df['draw_date'] = pd.to_datetime(existing_df['draw_date'], format='%d/%m/%Y', errors='coerce')
            
            # Get the latest draw date
            latest_date = existing_df['draw_date'].max()
            
            logger.info(f"Latest draw date in our data: {latest_date}")
            
            # Check if we need to update (if today is after the latest draw date)
            today = datetime.now().date()
            if latest_date.date() < today:
                logger.info("Data needs updating. Scraping latest results.")
                new_df = self.scrape_results(max_pages=5)  # Only scrape a few pages for updates
                
                # Combine new and existing data
                combined_df = pd.concat([new_df, existing_df])
                
                # Remove duplicates based on draw number
                combined_df = combined_df.drop_duplicates(subset=['draw_number'])
                
                # Sort by draw date (newest first)
                combined_df = combined_df.sort_values('draw_date', ascending=False)
                
                # Save to CSV
                combined_df.to_csv(self.csv_path, index=False)
                
                logger.info(f"Updated data with {len(new_df)} new results")
                
                return combined_df
            else:
                logger.info("Data is up to date. No need to scrape.")
                return existing_df
        except Exception as e:
            logger.error(f"Error reading existing data: {e}")
            logger.info("Falling back to scraping all results.")
            return self.scrape_results()


if __name__ == "__main__":
    # Test the scraper
    scraper = LottoScraper(data_dir='../data')
    results = scraper.update_results()
    print(f"Loaded {len(results)} lottery results")
    print(results.head())
