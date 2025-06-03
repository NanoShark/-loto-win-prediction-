import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime
import json
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from random import randint
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LottoPredictor:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, 'lotto_results.csv')
        self.excel_path = os.path.join(data_dir, 'previous_data.xlsx')
        self.predictions_dir = os.path.join(data_dir, 'predictions')
        
        # Create predictions directory if it doesn't exist
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Load data
        self.df = self.load_data()
        
        # Constants for Israeli Lotto
        self.regular_range = range(1, 71)  # Regular numbers are 1-70
        self.strong_range = range(1, 26)   # Strong numbers are 1-25
        self.num_regular = 6  # Number of regular numbers to select
        
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
        
        # Convert to Excel for the prediction algorithm
        df.to_excel(self.excel_path, index=False)
        
        return df
    
    def prepare_data_for_model(self):
        """
        Prepare data for the machine learning model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if Excel file exists
            if not os.path.exists(self.excel_path):
                if not os.path.exists(self.csv_path):
                    logger.error(f"No data files found at {self.csv_path} or {self.excel_path}")
                    return False
                else:
                    # Convert CSV to Excel if needed
                    df = pd.read_csv(self.csv_path, encoding='iso-8859-1')
                    df.to_excel(self.excel_path, index=False)
                    logger.info(f"Converted CSV to Excel at {self.excel_path}")
            return True
        except Exception as e:
            logger.error(f"Error preparing data for model: {e}")
            return False
    
    def get_prediction_stats(self):
        """
        Get statistics for prediction customization
        
        Returns:
            Dictionary with hot numbers, cold numbers, and due numbers
        """
        try:
            # Load data
            df = self.load_data()
            
            if df.empty:
                return {}
            
            # Get the last 30 draws
            recent_df = df.head(30)
            
            # Flatten all regular numbers
            all_numbers = []
            for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
                all_numbers.extend(recent_df[col].dropna().tolist())
            
            # Count frequencies
            number_counts = Counter(all_numbers)
            
            # Get hot numbers (top 10 most frequent)
            hot_numbers = [num for num, count in number_counts.most_common(10)]
            
            # Get cold numbers (10 least frequent from numbers that appeared at least once)
            all_appeared = set(number_counts.keys())
            cold_numbers = [num for num, count in sorted(number_counts.items(), key=lambda x: x[1])[:10]]
            
            # Get due numbers (numbers that haven't appeared in last 30 draws)
            all_possible = set(range(1, 71))
            due_numbers = list(all_possible - all_appeared)
            due_numbers.sort()
            
            # Get strong number stats
            strong_numbers = recent_df['strong_number'].dropna().tolist()
            strong_counts = Counter(strong_numbers)
            
            # Hot and cold strong numbers
            hot_strong = [num for num, count in strong_counts.most_common(5)]
            
            # All possible strong numbers
            all_possible_strong = set(range(1, 26))
            appeared_strong = set(strong_counts.keys())
            due_strong = list(all_possible_strong - appeared_strong)
            due_strong.sort()
            
            return {
                'hot_numbers': hot_numbers,
                'cold_numbers': cold_numbers,
                'due_numbers': due_numbers,
                'hot_strong': hot_strong,
                'due_strong': due_strong
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction stats: {e}")
            return {}
    
    def generate_prediction(self, num_sets=10, weight_recent=1.0, favor_hot=False, favor_due=False):
        """
        Generate lottery number predictions using Random Forest model
        
        Args:
            num_sets: Number of prediction sets to generate
            weight_recent: Weight factor for recent draws (1.0 = normal, >1.0 = more weight to recent draws)
            favor_hot: Whether to favor hot numbers in predictions
            favor_due: Whether to favor due numbers in predictions
            
        Returns:
            List of prediction sets
        """
        logger.info(f"Generating {num_sets} prediction sets with parameters: weight_recent={weight_recent}, favor_hot={favor_hot}, favor_due={favor_due}")
        
        # Prepare data for model
        if not self.prepare_data_for_model():
            logger.error("Failed to prepare data for model")
            return []
        
        # Load data from Excel file
        try:
            data = pd.read_excel(self.excel_path)
        except Exception as e:
            logger.error(f"Error loading data from Excel: {e}")
            return []
        
        # Apply weighting to recent draws if requested
        if weight_recent > 1.0:
            # Calculate weights based on recency
            num_rows = len(data)
            weights = np.ones(num_rows)
            
            # Apply exponential weighting - more recent draws get higher weights
            for i in range(num_rows):
                # Normalize position to 0-1 range (0 = oldest, 1 = newest)
                position = i / num_rows
                # Apply weight factor (higher weight_recent = steeper curve)
                weights[i] = 1 + (position * (weight_recent - 1))
            
            # Create weighted samples
            sample_indices = np.random.choice(
                range(num_rows),
                size=min(num_rows * 3, 1000),  # Use more samples for better representation
                replace=True,
                p=weights / weights.sum()
            )
            weighted_data = data.iloc[sample_indices].reset_index(drop=True)
        else:
            weighted_data = data
        
        # Get prediction stats if we're favoring hot or due numbers
        stats = self.get_prediction_stats() if (favor_hot or favor_due) else {}
        
        # Prepare features and target
        X = weighted_data[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']]
        y = weighted_data.iloc[:, 2:9]  # All number columns
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=1000, random_state=42)
        model.fit(X, y)
        
        # Generate predictions
        predictions = []
        i = 0
        
        while i < num_sets:
            try:
                # Generate random input features
                new_data = pd.DataFrame({
                    'number_1': [random.randint(1, 70)],
                    'number_2': [random.randint(1, 70)],
                    'number_3': [random.randint(1, 70)],
                    'number_4': [random.randint(1, 70)],
                    'number_5': [random.randint(1, 70)],
                    'number_6': [random.randint(1, 70)]
                })
                
                # Make prediction
                prediction_results = model.predict(new_data)
                
                # Get most likely set
                most_likely_set = prediction_results[0]
                
                # Round to nearest integers
                rounded_most_likely_set = [round(x) for x in most_likely_set]
                
                # Get regular numbers and ensure they are unique and within range
                regular_numbers = []
                for num in rounded_most_likely_set[:6]:
                    # Ensure number is within range 1-70
                    num = max(1, min(70, num))
                    if num not in regular_numbers:
                        regular_numbers.append(num)
                
                # If we're favoring hot numbers, include some hot numbers
                if favor_hot and 'hot_numbers' in stats and stats['hot_numbers']:
                    hot_nums = stats['hot_numbers']
                    # Replace some numbers with hot numbers (but keep at least half of the original prediction)
                    num_to_replace = min(2, len(regular_numbers) // 2)
                    for _ in range(num_to_replace):
                        if regular_numbers and hot_nums:  # Make sure we have numbers to replace
                            # Remove a random number from regular numbers
                            regular_numbers.pop(random.randrange(len(regular_numbers)))
                            # Add a hot number that's not already in the set
                            for hot_num in hot_nums:
                                if hot_num not in regular_numbers:
                                    regular_numbers.append(hot_num)
                                    break
                
                # If we're favoring due numbers, include some due numbers
                if favor_due and 'due_numbers' in stats and stats['due_numbers']:
                    due_nums = stats['due_numbers']
                    # Replace some numbers with due numbers (but keep at least half of the original prediction)
                    num_to_replace = min(2, len(regular_numbers) // 2)
                    for _ in range(num_to_replace):
                        if regular_numbers and due_nums:  # Make sure we have numbers to replace
                            # Remove a random number from regular numbers
                            regular_numbers.pop(random.randrange(len(regular_numbers)))
                            # Add a due number that's not already in the set
                            for due_num in due_nums:
                                if due_num not in regular_numbers:
                                    regular_numbers.append(due_num)
                                    break
                
                # If we don't have enough unique numbers, add some random ones
                while len(regular_numbers) < 6:
                    num = randint(1, 70)
                    if num not in regular_numbers:
                        regular_numbers.append(num)
                
                # Sort regular numbers
                regular_numbers.sort()
                
                # Get strong number and ensure it's within range 1-25
                strong_number = max(1, min(25, rounded_most_likely_set[6]))
                
                # If favoring hot or due for strong number
                if favor_hot and 'hot_strong' in stats and stats['hot_strong'] and random.random() < 0.5:
                    # 50% chance to use a hot strong number
                    strong_number = random.choice(stats['hot_strong'])
                elif favor_due and 'due_strong' in stats and stats['due_strong'] and random.random() < 0.5:
                    # 50% chance to use a due strong number
                    strong_number = random.choice(stats['due_strong'])
                
                predictions.append({
                    'regular_numbers': regular_numbers,
                    'strong_number': strong_number
                })
                
                logger.info(f"Generated prediction set {i+1}: {regular_numbers}, Strong: {strong_number}")
                i += 1
                
            except Exception as e:
                logger.error(f"Error in prediction iteration {i}: {e}")
                # Continue to next iteration
                i += 1
        
        return predictions
    
    def save_prediction(self, prediction):
        """
        Save prediction to file
        
        Args:
            prediction: Prediction data to save
            
        Returns:
            Path to the saved prediction file
        """
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create prediction data with metadata
        prediction_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_sets': prediction
        }
        
        # Save to JSON file
        prediction_file = os.path.join(self.predictions_dir, f'prediction_{timestamp}.json')
        
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=4)
        
        logger.info(f"Saved prediction to {prediction_file}")
        
        return prediction_file
    
    def get_latest_prediction(self):
        """
        Get the most recent prediction
        
        Returns:
            Dictionary containing the latest prediction data
        """
        # Get list of prediction files
        prediction_files = [f for f in os.listdir(self.predictions_dir) if f.startswith('prediction_') and f.endswith('.json')]
        
        if not prediction_files:
            logger.info("No prediction files found")
            return None
        
        # Sort by timestamp (newest first)
        prediction_files.sort(reverse=True)
        
        # Load the latest prediction
        latest_file = os.path.join(self.predictions_dir, prediction_files[0])
        
        with open(latest_file, 'r') as f:
            prediction_data = json.load(f)
        
        logger.info(f"Loaded latest prediction from {latest_file}")
        
        return prediction_data
    
    def run_prediction(self, num_sets=10, weight_recent=1.0, favor_hot=False, favor_due=False):
        """
        Run the prediction process
        
        Args:
            num_sets: Number of prediction sets to generate
            weight_recent: Weight factor for recent draws (1.0 = normal, >1.0 = more weight to recent draws)
            favor_hot: Whether to favor hot numbers in predictions
            favor_due: Whether to favor due numbers in predictions
            
        Returns:
            Dictionary containing the prediction data
        """
        logger.info(f"Starting prediction process with parameters: num_sets={num_sets}, weight_recent={weight_recent}, favor_hot={favor_hot}, favor_due={favor_due}")
        
        # Generate prediction with parameters
        prediction = self.generate_prediction(
            num_sets=num_sets,
            weight_recent=weight_recent,
            favor_hot=favor_hot,
            favor_due=favor_due
        )
        
        # Save prediction
        self.save_prediction(prediction)
        
        logger.info(f"Generated {len(prediction)} prediction sets")
        
        return prediction


if __name__ == "__main__":
    # Test the predictor
    predictor = LottoPredictor(data_dir='../data')
    prediction = predictor.run_prediction()
    
    print("Prediction sets:")
    for i, pred_set in enumerate(prediction, 1):
        print(f"Set {i}: Regular numbers: {pred_set['regular_numbers']}, Strong number: {pred_set['strong_number']}")
