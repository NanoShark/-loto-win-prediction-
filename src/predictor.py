import pandas as pd
import numpy as np
import random
import logging
import os
from datetime import datetime
import json
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LottoPredictor:
    def __init__(self, data_dir=None):
        # Use absolute path to data directory
        self.data_dir = data_dir if data_dir else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'data'
        )
        self.csv_path = os.path.join(self.data_dir, 'Lotto.csv')
        self.excel_path = os.path.join(self.data_dir, 'previous_data.xlsx')
        self.predictions_dir = os.path.join(self.data_dir, 'predictions')
        
        # Create predictions directory if it doesn't exist
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        logger.debug(f"Data directory: {self.data_dir}")
        logger.debug(f"CSV path: {self.csv_path}")
        logger.debug(f"Excel path: {self.excel_path}")
        
        # Load data
        self.df = self.load_data()
        
        # Constants for Israeli Lotto
        self.regular_range = range(1, 38)  # Regular numbers are 1-37
        self.strong_range = range(1, 8)    # Strong numbers are 1-7
        self.num_regular = 6  # Number of regular numbers to select
        
        # Initialize models
        self.regular_model = None
        self.strong_model = None
        
    def load_data(self):
        """Load lottery results from data file"""
        # Try CSV first
        if os.path.exists(self.csv_path):
            logger.info(f"Loading data from {self.csv_path}")
            try:
                # Read the CSV file with the correct encoding
                df = pd.read_csv(self.csv_path, encoding='iso-8859-1')
                
                # Rename columns to match the expected format
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
                
                # Remove rows with missing values
                df = df.dropna(subset=number_cols)
                
                # Sort by draw date (newest first)
                df = df.sort_values('draw_date', ascending=False)
                
                # Convert to Excel for compatibility
                df.to_excel(self.excel_path, index=False)
                
                logger.info(f"Loaded {len(df)} lottery results")
                return df
                
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return pd.DataFrame()
            
        # Fall back to Excel
        if os.path.exists(self.excel_path):
            logger.info(f"Loading data from {self.excel_path}")
            return pd.read_excel(self.excel_path)
            
        logger.error(f"No data files found. Checked:\n- {self.csv_path}\n- {self.excel_path}")
        return pd.DataFrame()
    
    def create_features(self, df):
        """Create additional features for the model"""
        features_df = df.copy()
        
        # Add statistical features
        regular_cols = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
        
        # Sum of numbers
        features_df['sum_numbers'] = features_df[regular_cols].sum(axis=1)
        
        # Average of numbers
        features_df['avg_numbers'] = features_df[regular_cols].mean(axis=1)
        
        # Standard deviation of numbers
        features_df['std_numbers'] = features_df[regular_cols].std(axis=1)
        
        # Number of odd numbers
        features_df['odd_count'] = features_df[regular_cols].apply(lambda row: sum(x % 2 == 1 for x in row), axis=1)
        
        # Number of even numbers
        features_df['even_count'] = features_df[regular_cols].apply(lambda row: sum(x % 2 == 0 for x in row), axis=1)
        
        # Range of numbers
        features_df['range_numbers'] = features_df[regular_cols].max(axis=1) - features_df[regular_cols].min(axis=1)
        
        # Add lag features (previous draw numbers)
        for i in range(1, 4):  # Last 3 draws
            for col in regular_cols + ['strong_number']:
                features_df[f'{col}_lag_{i}'] = features_df[col].shift(i)
        
        # Add frequency features
        features_df = self.add_frequency_features(features_df)
        
        return features_df
    
    def add_frequency_features(self, df):
        """Add frequency-based features"""
        # Calculate frequency of each number in the last 50 draws
        recent_draws = 50
        
        for i, row in df.iterrows():
            if i >= recent_draws:
                # Get the last 50 draws before this one
                recent_data = df.iloc[i-recent_draws:i]
                
                # Calculate frequency for each regular number
                all_numbers = []
                for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
                    all_numbers.extend(recent_data[col].tolist())
                
                number_freq = Counter(all_numbers)
                
                # Add frequency of current numbers
                for j, col in enumerate(['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']):
                    df.loc[i, f'{col}_frequency'] = number_freq.get(row[col], 0)
                
                # Strong number frequency
                strong_freq = Counter(recent_data['strong_number'].tolist())
                df.loc[i, 'strong_number_frequency'] = strong_freq.get(row['strong_number'], 0)
        
        return df
    
    def train_models(self, weight_recent=1.0):
        """Train separate models for regular numbers and strong numbers"""
        if self.df.empty:
            logger.error("No data available for training")
            return False
        
        logger.info("Training prediction models...")
        
        # Create features
        features_df = self.create_features(self.df)
        
        # Remove rows with NaN values (due to lag features)
        features_df = features_df.dropna()
        
        if len(features_df) < 20:
            logger.error("Not enough data for training")
            return False
        
        # Apply weighting to recent draws
        if weight_recent > 1.0:
            num_rows = len(features_df)
            sample_weights = np.ones(num_rows)
            
            for i in range(num_rows):
                # More recent draws (lower index) get higher weights
                position = (num_rows - i - 1) / num_rows
                sample_weights[i] = 1 + (position * (weight_recent - 1))
        else:
            sample_weights = None
        
        # Prepare features for regular numbers
        feature_cols = [col for col in features_df.columns if col not in 
                       ['draw_number', 'draw_date', 'number_1', 'number_2', 'number_3', 
                        'number_4', 'number_5', 'number_6', 'strong_number']]
        
        if not feature_cols:
            # Fallback to basic features
            feature_cols = ['sum_numbers', 'avg_numbers', 'std_numbers', 'odd_count', 'even_count']
            # Ensure these columns exist
            for col in feature_cols:
                if col not in features_df.columns:
                    feature_cols.remove(col)
        
        X = features_df[feature_cols]
        
        # Train regular numbers model
        y_regular = features_df[['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']]
        
        self.regular_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        if sample_weights is not None:
            self.regular_model.fit(X, y_regular, sample_weight=sample_weights)
        else:
            self.regular_model.fit(X, y_regular)
        
        # Train strong numbers model
        y_strong = features_df['strong_number']
        
        # Convert to classification problem (1-7 classes)
        y_strong = y_strong.astype(int) - 1  # Convert to 0-6 range
        
        self.strong_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        if sample_weights is not None:
            self.strong_model.fit(X, y_strong, sample_weight=sample_weights)
        else:
            self.strong_model.fit(X, y_strong)
        
        logger.info("Models trained successfully")
        return True
    
    def get_number_statistics(self, recent_draws=50):
        """Get statistical information about number frequencies"""
        if self.df.empty:
            return {}
        
        recent_data = self.df.head(recent_draws)
        
        # Regular numbers statistics
        all_regular = []
        for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
            all_regular.extend(recent_data[col].dropna().tolist())
        
        regular_counts = Counter(all_regular)
        
        # Hot numbers (most frequent)
        hot_numbers = [num for num, count in regular_counts.most_common(15)]
        
        # Cold numbers (least frequent among those that appeared)
        cold_numbers = [num for num, count in sorted(regular_counts.items(), key=lambda x: x[1])[:10]]
        
        # Due numbers (haven't appeared recently)
        appeared_regular = set(regular_counts.keys())
        due_numbers = [num for num in range(1, 38) if num not in appeared_regular]
        
        # Strong number statistics
        strong_counts = Counter(recent_data['strong_number'].dropna().tolist())
        hot_strong = [num for num, count in strong_counts.most_common(5)]
        
        appeared_strong = set(strong_counts.keys())
        due_strong = [num for num in range(1, 8) if num not in appeared_strong]
        
        return {
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'due_numbers': due_numbers,
            'hot_strong': hot_strong,
            'due_strong': due_strong,
            'regular_frequencies': dict(regular_counts),
            'strong_frequencies': dict(strong_counts)
        }
    
    def predict_strong_number(self, features, favor_hot=False, favor_due=False, stats=None):
        """Predict strong number with enforced distribution"""
        # First try to get historical distribution
        if not self.df.empty:
            strong_counts = self.df['strong_number'].value_counts(normalize=True)
            # Fill any missing numbers with minimum probability
            for num in range(1, 8):
                if num not in strong_counts:
                    strong_counts[num] = 0.01
            probs = strong_counts.sort_index().values
        else:
            # Fallback to uniform distribution
            probs = np.ones(7)/7
        
        # Make weighted random choice
        return np.random.choice(range(1,8), p=probs)
    
    def generate_prediction_set(self, favor_hot=False, favor_due=False, stats=None):
        """Generate a single prediction set"""
        if not self.regular_model or not self.strong_model:
            logger.error("Models not trained")
            return None
        
        # Create random features for prediction
        features = [
            random.uniform(50, 150),    # sum_numbers
            random.uniform(8, 25),      # avg_numbers
            random.uniform(5, 15),      # std_numbers
            random.randint(1, 6),       # odd_count
            random.randint(0, 5),       # even_count
            random.randint(10, 35),     # range_numbers
        ]
        
        # Add more features to match training data
        while len(features) < 20:  # Adjust based on actual feature count
            features.append(random.uniform(0, 1))
        
        # Predict regular numbers
        regular_pred = self.regular_model.predict([features])[0]
        
        # Process regular numbers
        regular_numbers = []
        for pred in regular_pred:
            num = max(1, min(37, round(pred)))
            if num not in regular_numbers:
                regular_numbers.append(num)
        
        # Apply preferences for regular numbers
        if favor_hot and stats and stats.get('hot_numbers'):
            # Replace some numbers with hot numbers
            for _ in range(min(2, len(regular_numbers))):
                if regular_numbers:
                    regular_numbers.pop()
                    hot_candidates = [n for n in stats['hot_numbers'] if n not in regular_numbers]
                    if hot_candidates:
                        regular_numbers.append(random.choice(hot_candidates))
        
        if favor_due and stats and stats.get('due_numbers'):
            # Replace some numbers with due numbers
            for _ in range(min(2, len(regular_numbers))):
                if regular_numbers:
                    regular_numbers.pop()
                    due_candidates = [n for n in stats['due_numbers'] if n not in regular_numbers]
                    if due_candidates:
                        regular_numbers.append(random.choice(due_candidates))
        
        # Ensure we have 6 unique numbers
        while len(regular_numbers) < 6:
            num = random.randint(1, 37)
            if num not in regular_numbers:
                regular_numbers.append(num)
        
        # Keep only 6 numbers
        regular_numbers = regular_numbers[:6]
        regular_numbers.sort()
        
        # Predict strong number
        strong_number = self.predict_strong_number(features, favor_hot, favor_due, stats)
        
        return {
            'regular_numbers': regular_numbers,
            'strong_number': strong_number
        }
    
    def generate_predictions(self, num_sets=10, weight_recent=1.0, favor_hot=False, favor_due=False):
        """Generate multiple prediction sets"""
        logger.info(f"Generating {num_sets} prediction sets")
        
        # Train models
        if not self.train_models(weight_recent=weight_recent):
            logger.error("Failed to train models")
            return []
        
        # Get statistics
        stats = self.get_number_statistics()
        
        # Generate predictions
        predictions = []
        for i in range(num_sets):
            try:
                pred_set = self.generate_prediction_set(favor_hot, favor_due, stats)
                if pred_set:
                    predictions.append(pred_set)
                    logger.info(f"Set {i+1}: {pred_set['regular_numbers']}, Strong: {pred_set['strong_number']}")
            except Exception as e:
                logger.error(f"Error generating prediction set {i+1}: {e}")
                continue
        
        return predictions
    
    def save_prediction(self, predictions):
        """Save predictions to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        prediction_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_sets': len(predictions),
            'prediction_sets': predictions,
            'statistics': self.get_number_statistics()
        }
        
        prediction_file = os.path.join(self.predictions_dir, f'prediction_{timestamp}.json')
        
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=4)
        
        logger.info(f"Saved {len(predictions)} predictions to {prediction_file}")
        return prediction_file
    
    def get_latest_prediction(self):
        """
        Get the most recent prediction from the predictions directory.
        
        Returns:
            Dictionary containing the latest prediction data, or None if no predictions found.
        """
        if not os.path.exists(self.predictions_dir):
            logger.error(f"Predictions directory not found: {self.predictions_dir}")
            return None

        prediction_files = [
            f for f in os.listdir(self.predictions_dir) 
            if f.startswith('prediction_') and f.endswith('.json')
        ]
        
        if not prediction_files:
            logger.warning("No prediction files available")
            return None
            
        # Sort by filename (timestamp) to get the latest
        prediction_files.sort(reverse=True)
        
        latest_file_path = os.path.join(self.predictions_dir, prediction_files[0])
        
        try:
            with open(latest_file_path, 'r') as f:
                prediction_data = json.load(f)
            logger.info(f"Loaded latest prediction from {latest_file_path}")
            return prediction_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {latest_file_path}: {e}")
        except FileNotFoundError:
            logger.error(f"Prediction file not found: {latest_file_path}")
        except Exception as e:
            logger.error(f"Unexpected error loading {latest_file_path}: {e}")
        return None
    
    def run_prediction(self, num_sets=10, weight_recent=1.5, favor_hot=False, favor_due=False):
        """Run the complete prediction process"""
        logger.info(f"Starting prediction process with {num_sets} sets")
        
        # Generate predictions
        predictions = self.generate_predictions(
            num_sets=num_sets,
            weight_recent=weight_recent,
            favor_hot=favor_hot,
            favor_due=favor_due
        )
        
        if not predictions:
            logger.error("No predictions generated")
            return []
        
        # Save predictions
        self.save_prediction(predictions)
        
        logger.info(f"Successfully generated {len(predictions)} prediction sets")
        return predictions


if __name__ == "__main__":
    # Test the predictor
    predictor = LottoPredictor(data_dir='../data')
    
    # Generate predictions with different strategies
    print("=== Standard Prediction ===")
    standard_pred = predictor.run_prediction(num_sets=3)
    
    print("\n=== Hot Numbers Strategy ===")
    hot_pred = predictor.run_prediction(num_sets=3, favor_hot=True)
    
    print("\n=== Due Numbers Strategy ===") 
    due_pred = predictor.run_prediction(num_sets=3, favor_due=True)
    
    print("\n=== Recent Weight Strategy ===")
    recent_pred = predictor.run_prediction(num_sets=3, weight_recent=2.0)
