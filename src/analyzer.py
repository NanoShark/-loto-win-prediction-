import pandas as pd
import numpy as np
from collections import Counter
import logging
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LottoAnalyzer:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, 'lotto_results.csv')
        self.stats_dir = os.path.join(data_dir, 'stats')
        
        # Create stats directory if it doesn't exist
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Load data
        self.df = self.load_data()
        
        # Constants for Israeli Lotto
        self.regular_range = range(1, 38)  # Regular numbers are 1-37
        self.strong_range = range(1, 9)    # Strong numbers are 1-8
        
    def load_data(self):
        """
        Load lottery results from CSV file
        
        Returns:
            DataFrame containing the lottery results
        """
        if not os.path.exists(self.csv_path):
            logger.error(f"Data file not found at {self.csv_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading lottery results from {self.csv_path}")
        
        # Read the CSV file with the correct encoding
        df = pd.read_csv(self.csv_path, encoding='iso-8859-1')
        
        # Rename columns to match the expected format
        # The first column is the draw number, second is the date, then 6 regular numbers, then the strong number
        column_mapping = {
            df.columns[0]: 'draw_number',
            df.columns[1]: 'draw_date',
            df.columns[2]: 'number_1',
            df.columns[3]: 'number_2',
            df.columns[4]: 'number_3',
            df.columns[5]: 'number_4',
            df.columns[6]: 'number_5',
            df.columns[7]: 'number_6',
            df.columns[8]: 'strong_number'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert date strings to datetime objects
        df['draw_date'] = pd.to_datetime(df['draw_date'], format='%d/%m/%Y', errors='coerce')
        
        # Ensure number columns are integers
        number_cols = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'strong_number']
        for col in number_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Sort by draw date (newest first)
        df = df.sort_values('draw_date', ascending=False)
        
        return df
    
    def get_number_frequency(self, time_period=None):
        """
        Calculate the frequency of each regular number
        
        Args:
            time_period: Optional time period to filter results (in days)
            
        Returns:
            Dictionary with number frequencies
        """
        if self.df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Filter by time period if specified
        filtered_df = self.df
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            filtered_df = self.df[self.df['draw_date'] >= cutoff_date]
        
        # Extract all regular numbers
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(filtered_df[f'number_{i}'].tolist())
        
        # Count frequencies
        frequency = Counter(all_numbers)
        
        # Ensure all possible numbers are included (even if frequency is 0)
        for num in self.regular_range:
            if num not in frequency:
                frequency[num] = 0
        
        return dict(sorted(frequency.items()))
    
    def get_strong_number_frequency(self, time_period=None):
        """
        Calculate the frequency of each strong number
        
        Args:
            time_period: Optional time period to filter results (in days)
            
        Returns:
            Dictionary with strong number frequencies
        """
        if self.df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Filter by time period if specified
        filtered_df = self.df
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            filtered_df = self.df[self.df['draw_date'] >= cutoff_date]
        
        # Count frequencies
        frequency = Counter(filtered_df['strong_number'])
        
        # Ensure all possible numbers are included (even if frequency is 0)
        for num in self.strong_range:
            if num not in frequency:
                frequency[num] = 0
        
        return dict(sorted(frequency.items()))
    
    def get_hot_numbers(self, count=10, time_period=None):
        """
        Get the most frequently drawn regular numbers
        
        Args:
            count: Number of hot numbers to return
            time_period: Optional time period to filter results (in days)
            
        Returns:
            List of hot numbers
        """
        frequency = self.get_number_frequency(time_period)
        hot_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:count]
        return [num for num, freq in hot_numbers]
    
    def get_cold_numbers(self, count=10, time_period=None):
        """
        Get the least frequently drawn regular numbers
        
        Args:
            count: Number of cold numbers to return
            time_period: Optional time period to filter results (in days)
            
        Returns:
            List of cold numbers
        """
        frequency = self.get_number_frequency(time_period)
        cold_numbers = sorted(frequency.items(), key=lambda x: x[1])[:count]
        return [num for num, freq in cold_numbers]
    
    def get_hot_strong_numbers(self, count=3, time_period=None):
        """
        Get the most frequently drawn strong numbers
        
        Args:
            count: Number of hot strong numbers to return
            time_period: Optional time period to filter results (in days)
            
        Returns:
            List of hot strong numbers
        """
        frequency = self.get_strong_number_frequency(time_period)
        hot_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:count]
        return [num for num, freq in hot_numbers]
    
    def get_cold_strong_numbers(self, count=3, time_period=None):
        """
        Get the least frequently drawn strong numbers
        
        Args:
            count: Number of cold strong numbers to return
            time_period: Optional time period to filter results (in days)
            
        Returns:
            List of cold strong numbers
        """
        frequency = self.get_strong_number_frequency(time_period)
        cold_numbers = sorted(frequency.items(), key=lambda x: x[1])[:count]
        return [num for num, freq in cold_numbers]
    
    def get_number_pairs(self, time_period=None, top_n=10):
        """
        Find the most common pairs of numbers that appear together
        
        Args:
            time_period: Optional time period to filter results (in days)
            top_n: Number of top pairs to return
            
        Returns:
            Dictionary with pairs and their frequencies
        """
        if self.df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Filter by time period if specified
        filtered_df = self.df
        if time_period:
            cutoff_date = datetime.now() - timedelta(days=time_period)
            filtered_df = self.df[self.df['draw_date'] >= cutoff_date]
        
        # Extract all combinations of pairs
        pairs = []
        for _, row in filtered_df.iterrows():
            numbers = [row[f'number_{i}'] for i in range(1, 7)]
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    # Ensure the pair is ordered (smaller number first)
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    pairs.append(pair)
        
        # Count frequencies
        pair_frequency = Counter(pairs)
        
        # Get top N pairs
        top_pairs = pair_frequency.most_common(top_n)
        
        return {f"{pair[0]}-{pair[1]}": freq for pair, freq in top_pairs}
    
    def get_due_numbers(self, draws_threshold=10):
        """
        Find numbers that haven't appeared in recent draws
        
        Args:
            draws_threshold: Number of recent draws to check
            
        Returns:
            List of due numbers
        """
        if self.df.empty or len(self.df) < draws_threshold:
            logger.error("Not enough data available for analysis")
            return []
        
        # Get the most recent draws
        recent_draws = self.df.head(draws_threshold)
        
        # Extract all numbers from recent draws
        recent_numbers = []
        for i in range(1, 7):
            recent_numbers.extend(recent_draws[f'number_{i}'].tolist())
        
        # Find numbers that haven't appeared in recent draws
        due_numbers = [num for num in self.regular_range if num not in recent_numbers]
        
        return due_numbers
    
    def get_due_strong_numbers(self, draws_threshold=10):
        """
        Find strong numbers that haven't appeared in recent draws
        
        Args:
            draws_threshold: Number of recent draws to check
            
        Returns:
            List of due strong numbers
        """
        if self.df.empty or len(self.df) < draws_threshold:
            logger.error("Not enough data available for analysis")
            return []
        
        # Get the most recent draws
        recent_draws = filtered_df.head(draws_threshold)
        
        # Extract all strong numbers from recent draws
        recent_strong_numbers = recent_draws['strong_number'].tolist()
        
        # Find strong numbers that haven't appeared in recent draws
        due_strong_numbers = [num for num in self.strong_range if num not in recent_strong_numbers]
        
        return due_strong_numbers
    
    def generate_statistics(self, days=None):
        """
        Generate statistics about lottery numbers
        
        Args:
            days: Optional number of days to filter data (None for all time)
            
        Returns:
            Dictionary of statistics
        """
        logger.info(f"Generating lottery statistics for time range: {days if days else 'all time'} days")
        
        # Load data if not already loaded
        if self.df is None or self.df.empty:
            self.df = self.load_data()
        
        if self.df.empty:
            logger.error("No data available for statistics generation")
            return {}
        
        # Create stats directory if it doesn't exist
        os.makedirs(os.path.join(self.data_dir, 'stats'), exist_ok=True)
        
        # Path to save statistics
        stats_file = os.path.join(self.data_dir, 'stats', 'statistics.csv')
        
        # Determine time ranges based on the days parameter
        if days is not None:
            # If specific days are requested, adjust the time ranges
            time_ranges = {
                'all_time': None,  # Always include all time
                'specified': days,  # The requested time range
                'last_quarter': min(90, days) if days > 30 else 90,  # Include quarterly if days > 30
                'last_month': min(30, days) if days > 0 else 30  # Include monthly if days > 0
            }
        else:
            # Default time ranges
            time_ranges = {
                'all_time': None,
                'last_year': 365,
                'last_quarter': 90,
                'last_month': 30
            }
        
        # Calculate statistics
        stats = {
            'total_draws': len(self.df),
            'date_range': {
                'start': self.df['draw_date'].min().strftime('%Y-%m-%d'),
                'end': self.df['draw_date'].max().strftime('%Y-%m-%d')
            },
            'frequency': {},
            'strong_frequency': {},
            'hot_numbers': {},
            'cold_numbers': {},
            'hot_strong_numbers': {},
            'cold_strong_numbers': {}
        }
        
        # Generate statistics for each time range
        for key, range_days in time_ranges.items():
            stats['frequency'][key] = self.get_number_frequency(days=range_days)
            stats['strong_frequency'][key] = self.get_strong_number_frequency(days=range_days)
            stats['hot_numbers'][key] = self.get_hot_numbers(days=range_days)
            stats['cold_numbers'][key] = self.get_cold_numbers(days=range_days)
            stats['hot_strong_numbers'][key] = self.get_hot_strong_numbers(days=range_days)
            stats['cold_strong_numbers'][key] = self.get_cold_strong_numbers(days=range_days)
        
        # Add common pairs and due numbers
        stats['common_pairs'] = self.get_number_pairs(top_n=15, days=days)
        stats['due_numbers'] = self.get_due_numbers(days=days)
        stats['due_strong_numbers'] = self.get_due_strong_numbers(days=days)
        
        # Save statistics to CSV
        stats_df = pd.DataFrame({
            'statistic': list(stats.keys()),
            'value': [str(v) for v in stats.values()]
        })
        stats_df.to_csv(stats_file, index=False)
        
        logger.info(f"Statistics saved to {stats_file}")
        
        return stats
    
    def plot_number_frequency(self, time_period=None, save_path=None):
        """
        Plot the frequency of regular numbers
        
        Args:
            time_period: Optional time period to filter results (in days)
            save_path: Path to save the plot (if None, plot is displayed)
        """
        frequency = self.get_number_frequency(time_period)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x=list(frequency.keys()), y=list(frequency.values()))
        
        title = "Frequency of Regular Numbers"
        if time_period:
            title += f" (Last {time_period} days)"
        
        plt.title(title)
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved number frequency plot to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def plot_strong_number_frequency(self, time_period=None, save_path=None):
        """
        Plot the frequency of strong numbers
        
        Args:
            time_period: Optional time period to filter results (in days)
            save_path: Path to save the plot (if None, plot is displayed)
        """
        frequency = self.get_strong_number_frequency(time_period)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(frequency.keys()), y=list(frequency.values()))
        
        title = "Frequency of Strong Numbers"
        if time_period:
            title += f" (Last {time_period} days)"
        
        plt.title(title)
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved strong number frequency plot to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def generate_visualizations(self):
        """
        Generate and save various visualizations
        """
        if self.df.empty:
            logger.error("No data available for analysis")
            return
        
        # Create visualizations directory if it doesn't exist
        vis_dir = os.path.join(self.stats_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot number frequencies
        self.plot_number_frequency(
            save_path=os.path.join(vis_dir, 'number_frequency_all_time.png')
        )
        self.plot_number_frequency(
            time_period=365,
            save_path=os.path.join(vis_dir, 'number_frequency_last_year.png')
        )
        self.plot_number_frequency(
            time_period=30,
            save_path=os.path.join(vis_dir, 'number_frequency_last_month.png')
        )
        
        # Plot strong number frequencies
        self.plot_strong_number_frequency(
            save_path=os.path.join(vis_dir, 'strong_number_frequency_all_time.png')
        )
        self.plot_strong_number_frequency(
            time_period=365,
            save_path=os.path.join(vis_dir, 'strong_number_frequency_last_year.png')
        )
        self.plot_strong_number_frequency(
            time_period=30,
            save_path=os.path.join(vis_dir, 'strong_number_frequency_last_month.png')
        )
        
        logger.info(f"Generated visualizations and saved to {vis_dir}")


if __name__ == "__main__":
    # Test the analyzer
    analyzer = LottoAnalyzer(data_dir='../data')
    stats = analyzer.generate_statistics()
    analyzer.generate_visualizations()
    print("Analysis complete!")
