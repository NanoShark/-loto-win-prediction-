import json
import logging
import os
import random
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/predictor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LottoPredictor:
    def __init__(self, data_dir=None):
        # Use absolute path to data directory
        self.data_dir = (
            data_dir
            if data_dir
            else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
            )
        )
        self.csv_path = os.path.join(self.data_dir, "Lotto.csv")
        self.excel_path = os.path.join(self.data_dir, "previous_data.xlsx")
        self.predictions_dir = os.path.join(self.data_dir, "predictions")

        # Create predictions directory if it doesn't exist
        os.makedirs(self.predictions_dir, exist_ok=True)

        logger.debug(f"Data directory: {self.data_dir}")
        logger.debug(f"CSV path: {self.csv_path}")
        logger.debug(f"Excel path: {self.excel_path}")

        # Load data
        self.df = self.load_data()

        # Constants for Israeli Lotto
        self.regular_range = range(1, 38)  # Regular numbers are 1-37
        self.strong_range = range(1, 8)  # Strong numbers are 1-7
        self.num_regular = 6  # Number of regular numbers to select

        # Initialize models
        self.regular_model = None
        self.strong_model = None

    def load_data(self):
        """
        Load lottery results from data file with validation

        Returns:
            DataFrame: Loaded and validated lottery data, or empty DataFrame if loading fails
        """

        def _validate_dataframe(df):
            """Validate the structure and content of the loaded DataFrame"""
            required_columns = [
                "draw_number",
                "draw_date",
                "number_1",
                "number_2",
                "number_3",
                "number_4",
                "number_5",
                "number_6",
                "strong_number",
            ]

            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for empty DataFrame
            if df.empty:
                raise ValueError("No data available in the loaded file")

            return True

        # Try CSV first
        if os.path.exists(self.csv_path):
            try:
                logger.info(f"Loading data from {self.csv_path}")

                # Read CSV with error handling for different encodings
                try:
                    df = pd.read_csv(self.csv_path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(self.csv_path, encoding="iso-8859-1")

                if df.empty:
                    logger.warning("CSV file is empty")
                    raise ValueError("Empty CSV file")

                # Check if we have enough columns
                if len(df.columns) < 9:
                    raise ValueError(
                        f"Expected at least 9 columns, got {len(df.columns)}"
                    )

                # Rename columns to match the expected format
                column_mapping = {
                    df.columns[0]: "draw_number",
                    df.columns[1]: "draw_date",
                    df.columns[2]: "number_1",
                    df.columns[3]: "number_2",
                    df.columns[4]: "number_3",
                    df.columns[5]: "number_4",
                    df.columns[6]: "number_5",
                    df.columns[7]: "number_6",
                    df.columns[8]: "strong_number",
                }

                df = df.rename(columns=column_mapping)

                # Validate data types and convert
                df["draw_date"] = pd.to_datetime(
                    df["draw_date"], format="%d/%m/%Y", errors="coerce"
                )

                # Ensure number columns are integers
                number_cols = [
                    "number_1",
                    "number_2",
                    "number_3",
                    "number_4",
                    "number_5",
                    "number_6",
                    "strong_number",
                ]
                for col in number_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

                # Remove rows with missing values
                initial_count = len(df)
                df = df.dropna(subset=number_cols + ["draw_date"])
                if len(df) < initial_count:
                    logger.warning(
                        f"Dropped {initial_count - len(df)} rows with missing values"
                    )

                if df.empty:
                    raise ValueError("No valid data remaining after cleaning")

                # Sort by draw date (newest first)
                df = df.sort_values("draw_date", ascending=False)

                # Validate data structure
                _validate_dataframe(df)

                # Save as Excel for future use
                try:
                    df.to_excel(self.excel_path, index=False)
                    logger.info(f"Saved backup to {self.excel_path}")
                except Exception as e:
                    logger.warning(f"Failed to save Excel backup: {e}")

                logger.info(f"Successfully loaded {len(df)} lottery results from CSV")
                return df

            except Exception as e:
                logger.error(f"Error loading CSV data: {e}")
                logger.info("Attempting to load from Excel backup...")
        else:
            logger.warning(f"CSV file not found: {self.csv_path}")

        # Fall back to Excel if CSV loading fails
        if os.path.exists(self.excel_path):
            try:
                logger.info(f"Loading data from Excel backup: {self.excel_path}")
                df = pd.read_excel(self.excel_path)
                _validate_dataframe(df)
                logger.info(
                    f"Successfully loaded {len(df)} lottery results from Excel backup"
                )
                return df
            except Exception as e:
                logger.error(f"Error loading Excel data: {e}")
        else:
            logger.warning(f"Excel backup not found: {self.excel_path}")

        logger.error("Failed to load data from any source")
        return pd.DataFrame()

    def create_features(self, df):
        """Create additional features for the model"""
        features_df = df.copy()

        # Add statistical features
        regular_cols = [
            "number_1",
            "number_2",
            "number_3",
            "number_4",
            "number_5",
            "number_6",
        ]

        # Sum of numbers
        features_df["sum_numbers"] = features_df[regular_cols].sum(axis=1)

        # Average of numbers
        features_df["avg_numbers"] = features_df[regular_cols].mean(axis=1)

        # Standard deviation of numbers
        features_df["std_numbers"] = features_df[regular_cols].std(axis=1)

        # Number of odd numbers
        features_df["odd_count"] = features_df[regular_cols].apply(
            lambda row: sum(x % 2 == 1 for x in row), axis=1
        )

        # Number of even numbers
        features_df["even_count"] = features_df[regular_cols].apply(
            lambda row: sum(x % 2 == 0 for x in row), axis=1
        )

        # Range of numbers
        features_df["range_numbers"] = features_df[regular_cols].max(
            axis=1
        ) - features_df[regular_cols].min(axis=1)

        # Add lag features (previous draw numbers)
        for i in range(1, 4):  # Last 3 draws
            for col in regular_cols + ["strong_number"]:
                features_df[f"{col}_lag_{i}"] = features_df[col].shift(i)

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
                recent_data = df.iloc[i - recent_draws : i]

                # Calculate frequency for each regular number
                all_numbers = []
                for col in [
                    "number_1",
                    "number_2",
                    "number_3",
                    "number_4",
                    "number_5",
                    "number_6",
                ]:
                    all_numbers.extend(recent_data[col].tolist())

                number_freq = Counter(all_numbers)

                # Add frequency of current numbers
                for j, col in enumerate(
                    [
                        "number_1",
                        "number_2",
                        "number_3",
                        "number_4",
                        "number_5",
                        "number_6",
                    ]
                ):
                    df.loc[i, f"{col}_frequency"] = number_freq.get(row[col], 0)

                # Strong number frequency
                strong_freq = Counter(recent_data["strong_number"].tolist())
                df.loc[i, "strong_number_frequency"] = strong_freq.get(
                    row["strong_number"], 0
                )

        return df

    def train_models(self, weight_recent=1.0):
        """
        Train separate models for regular numbers and strong numbers.

        Args:
            weight_recent (float): Weight to apply to recent draws (1.0 = equal weight)

        Returns:
            bool: True if models were trained successfully, False otherwise
        """
        # Validate input parameters
        if not isinstance(weight_recent, (int, float)) or weight_recent <= 0:
            logger.warning(f"Invalid weight_recent: {weight_recent}, using 1.0")
            weight_recent = 1.0

        # Check if we have enough data
        min_samples = 10  # Minimum number of samples required for training
        if self.df is None or len(self.df) < min_samples:
            logger.error(
                f"Not enough data for training. Need at least {min_samples} samples, got {len(self.df) if self.df is not None else 0}"
            )
            return False

        try:
            logger.info("Starting model training...")

            # Create features
            logger.debug("Creating features...")
            try:
                features_df = self.create_features(self.df)
                if features_df is None or features_df.empty:
                    logger.error("Feature creation failed - no features generated")
                    return False
            except Exception as e:
                logger.error(f"Error creating features: {e}")
                return False

            # Remove rows with NaN values (due to lag features)
            initial_count = len(features_df)
            features_df = features_df.dropna()

            if len(features_df) < min_samples:
                logger.error(
                    f"Not enough valid data after cleaning. Need {min_samples} samples, got {len(features_df)}"
                )
                return False
            elif len(features_df) < initial_count:
                logger.warning(
                    f"Dropped {initial_count - len(features_df)} rows with missing values"
                )

            # Apply weighting to recent draws if specified
            if weight_recent > 1.0:
                num_rows = len(features_df)
                sample_weights = np.linspace(
                    weight_recent, 1.0, num_rows
                )  # Linear decay
                logger.debug(f"Applied sample weights from {weight_recent:.2f} to 1.0")
            else:
                sample_weights = None

            # Prepare features for training
            feature_cols = [
                col
                for col in features_df.columns
                if col
                not in [
                    "draw_number",
                    "draw_date",
                    "number_1",
                    "number_2",
                    "number_3",
                    "number_4",
                    "number_5",
                    "number_6",
                    "strong_number",
                ]
            ]

            if not feature_cols:
                # Fallback to basic features
                feature_cols = [
                    "sum_numbers",
                    "avg_numbers",
                    "std_numbers",
                    "odd_count",
                    "even_count",
                ]
                feature_cols = [
                    col for col in feature_cols if col in features_df.columns
                ]

                if not feature_cols:
                    logger.error("No valid features found for training")
                    return False

            X = features_df[feature_cols]

            # Train regular numbers model
            try:
                logger.debug("Training regular numbers model...")
                y_regular = features_df[
                    [
                        "number_1",
                        "number_2",
                        "number_3",
                        "number_4",
                        "number_5",
                        "number_6",
                    ]
                ]

                # Clear any existing model to free memory
                if hasattr(self, "regular_model"):
                    del self.regular_model
                    import gc

                    gc.collect()

                self.regular_model = RandomForestRegressor(
                    n_estimators=100,  # Reduced from 200 for faster training
                    max_depth=8,  # Reduced from 10 to prevent overfitting
                    random_state=42,
                    n_jobs=-1,
                    verbose=1,
                )

                if sample_weights is not None:
                    self.regular_model.fit(X, y_regular, sample_weight=sample_weights)
                else:
                    self.regular_model.fit(X, y_regular)

                logger.debug("Regular numbers model trained successfully")

            except Exception as e:
                logger.error(f"Error training regular numbers model: {e}")
                self.regular_model = None
                return False

            # Train strong numbers model
            try:
                logger.debug("Training strong numbers model...")
                y_strong = features_df["strong_number"]

                # Convert to classification problem (1-7 classes -> 0-6 for sklearn)
                y_strong = y_strong.astype(int) - 1

                # Clear any existing model to free memory
                if hasattr(self, "strong_model"):
                    del self.strong_model
                    gc.collect()

                self.strong_model = RandomForestClassifier(
                    n_estimators=100,  # Reduced from 200 for faster training
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1,
                )

                if sample_weights is not None:
                    self.strong_model.fit(X, y_strong, sample_weight=sample_weights)
                else:
                    self.strong_model.fit(X, y_strong)

                logger.debug("Strong numbers model trained successfully")

            except Exception as e:
                logger.error(f"Error training strong numbers model: {e}")
                self.strong_model = None
                # Don't fail completely if only strong number model fails

            # If we got here, at least the regular model was trained successfully
            logger.info("Model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")
            logger.exception("Training error details:")
            return False
        finally:
            # Clean up large intermediate variables
            if "features_df" in locals():
                del features_df
            if "X" in locals():
                del X
            if "y_regular" in locals():
                del y_regular
            if "y_strong" in locals():
                del y_strong
            gc.collect()

    def get_number_statistics(self, recent_draws=50):
        """Get statistical information about number frequencies"""
        if self.df.empty:
            return {}

        recent_data = self.df.head(recent_draws)

        # Regular numbers statistics
        all_regular = []
        for col in [
            "number_1",
            "number_2",
            "number_3",
            "number_4",
            "number_5",
            "number_6",
        ]:
            all_regular.extend(recent_data[col].dropna().tolist())

        regular_counts = Counter(all_regular)

        # Hot numbers (most frequent)
        hot_numbers = [num for num, count in regular_counts.most_common(15)]

        # Cold numbers (least frequent among those that appeared)
        cold_numbers = [
            num
            for num, count in sorted(regular_counts.items(), key=lambda x: x[1])[:10]
        ]

        # Due numbers (haven't appeared recently)
        appeared_regular = set(regular_counts.keys())
        due_numbers = [num for num in range(1, 38) if num not in appeared_regular]

        # Strong number statistics
        strong_counts = Counter(recent_data["strong_number"].dropna().tolist())
        hot_strong = [num for num, count in strong_counts.most_common(5)]

        appeared_strong = set(strong_counts.keys())
        due_strong = [num for num in range(1, 8) if num not in appeared_strong]

        return {
            "hot_numbers": hot_numbers,
            "cold_numbers": cold_numbers,
            "due_numbers": due_numbers,
            "hot_strong": hot_strong,
            "due_strong": due_strong,
            "regular_frequencies": dict(regular_counts),
            "strong_frequencies": dict(strong_counts),
        }

    def predict_strong_number(
        self, features, favor_hot=False, favor_due=False, stats=None
    ):
        """Predict strong number with enforced distribution"""
        # First try to get historical distribution
        if not self.df.empty:
            strong_counts = self.df["strong_number"].value_counts(normalize=True)
            # Fill any missing numbers with minimum probability
            for num in range(1, 8):
                if num not in strong_counts:
                    strong_counts[num] = 0.01
            probs = strong_counts.sort_index().values
        else:
            # Fallback to uniform distribution
            probs = np.ones(7) / 7

        # Make weighted random choice
        return np.random.choice(range(1, 8), p=probs)

    def generate_prediction_set(self, favor_hot=False, favor_due=False, stats=None):
        """Generate a single prediction set"""
        try:
            if not self.regular_model or not self.strong_model:
                logger.error("Models not trained")
                return None

            # Create random features for prediction
            features = [
                random.uniform(50, 150),  # sum_numbers
                random.uniform(8, 25),  # avg_numbers
                random.uniform(5, 15),  # std_numbers
                random.randint(1, 6),  # odd_count
                random.randint(0, 5),  # even_count
                random.randint(10, 35),  # range_numbers
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
            if favor_hot and stats and stats.get("hot_numbers"):
                # Replace some numbers with hot numbers
                for _ in range(min(2, len(regular_numbers))):
                    if regular_numbers:
                        regular_numbers.pop()
                        hot_candidates = [
                            n for n in stats["hot_numbers"] if n not in regular_numbers
                        ]
                        if hot_candidates:
                            regular_numbers.append(random.choice(hot_candidates))

            if favor_due and stats and stats.get("due_numbers"):
                # Replace some numbers with due numbers
                for _ in range(min(2, len(regular_numbers))):
                    if regular_numbers:
                        regular_numbers.pop()
                        due_candidates = [
                            n for n in stats["due_numbers"] if n not in regular_numbers
                        ]
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
            strong_number = self.predict_strong_number(
                features, favor_hot, favor_due, stats
            )

            return {"regular_numbers": regular_numbers, "strong_number": strong_number}
        except Exception as e:
            logger.error(f"Error generating prediction set: {e}")
            logger.warning("Falling back to statistical prediction")
            return self._statistical_prediction(favor_hot, favor_due, stats)

    def _statistical_prediction(
        self,
        num_sets=1,
        weight_recent=1.0,
        favor_hot=False,
        favor_due=False,
        stats=None,
    ):
        """
        Fallback prediction method using statistical analysis when ML models fail.

        Args:
            num_sets (int): Number of prediction sets to generate
            weight_recent (float): Not used in this method, kept for API compatibility
            favor_hot (bool): Whether to favor hot numbers
            favor_due (bool): Whether to favor due numbers
            stats (dict): Pre-computed statistics (optional)

        Returns:
            list: List of prediction sets
        """
        logger.info("Using statistical prediction method")
        predictions = []

        # Get or compute statistics if not provided
        if stats is None:
            try:
                stats = self.get_number_statistics()
            except Exception as e:
                logger.warning(f"Could not get number statistics: {e}")
                stats = {}

        # Determine regular number candidates
        if favor_hot and "hot_numbers" in stats and stats["hot_numbers"]:
            regular_candidates = stats["hot_numbers"]
            logger.debug("Using hot numbers for regular numbers")
        elif favor_due and "due_numbers" in stats and stats["due_numbers"]:
            regular_candidates = stats["due_numbers"]
            logger.debug("Using due numbers for regular numbers")
        else:
            regular_candidates = list(range(1, 38))
            logger.debug("Using all possible regular numbers")

        # Determine strong number candidates
        if favor_hot and "hot_strong" in stats and stats["hot_strong"]:
            strong_candidates = stats["hot_strong"]
            logger.debug("Using hot numbers for strong number")
        elif favor_due and "due_strong" in stats and stats["due_strong"]:
            strong_candidates = stats["due_strong"]
            logger.debug("Using due numbers for strong number")
        else:
            strong_candidates = list(range(1, 8))
            logger.debug("Using all possible strong numbers")

        # Generate the requested number of prediction sets
        for _ in range(num_sets):
            try:
                # Ensure we have enough candidates
                if len(regular_candidates) < 6:
                    logger.warning(
                        "Not enough regular number candidates, using all numbers"
                    )
                    regular_candidates = list(range(1, 38))

                # Select 6 unique regular numbers
                regular_numbers = sorted(
                    random.sample(regular_candidates, min(6, len(regular_candidates)))
                )

                # Select 1 strong number
                strong_number = random.choice(strong_candidates)

                prediction = {
                    "regular_numbers": regular_numbers,
                    "strong_number": strong_number,
                    "method": "statistical",
                    "strategy": "hot"
                    if favor_hot
                    else "due"
                    if favor_due
                    else "random",
                }

                if self._validate_prediction(prediction):
                    predictions.append(prediction)
                else:
                    logger.warning("Generated invalid prediction, retrying...")

            except Exception as e:
                logger.error(f"Error in statistical prediction: {e}")
                # If we can't generate predictions, return what we have
                if len(predictions) > 0:
                    break

        logger.info(f"Generated {len(predictions)} statistical prediction sets")
        return predictions

    def _validate_prediction(self, prediction):
        """
        Validate that a prediction set is valid.

        Args:
            prediction (dict): Prediction set to validate

        Returns:
            bool: True if the prediction is valid, False otherwise
        """
        try:
            if not isinstance(prediction, dict):
                return False

            # Check required keys
            if "regular_numbers" not in prediction or "strong_number" not in prediction:
                return False

            regular = prediction["regular_numbers"]
            strong = prediction["strong_number"]

            # Check types
            if not isinstance(regular, (list, tuple)) or not isinstance(
                strong, (int, np.integer)
            ):
                return False

            # Check regular numbers
            if len(regular) != 6:
                return False

            # Check for duplicates
            if len(set(regular)) != 6:
                return False

            # Check number ranges
            if not all(1 <= num <= 37 for num in regular):
                return False

            # Check strong number range
            if not 1 <= strong <= 7:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating prediction: {e}")
            return False

    def run_prediction_process(
        self, num_sets=1, weight_recent=1.0, favor_hot=False, favor_due=False
    ):
        """
        Run the complete prediction process with fallback mechanisms.

        Args:
            num_sets (int): Number of prediction sets to generate (1-100)
            weight_recent (float): Weight to give to recent draws (1.0 = equal weight)
            favor_hot (bool): Whether to favor hot numbers in predictions
            favor_due (bool): Whether to favor due numbers in predictions

        Returns:
            list: List of prediction sets, each containing 'regular_numbers' and 'strong_number'
        """
        # Input validation
        try:
            num_sets = int(num_sets)
            if num_sets < 1 or num_sets > 100:
                logger.warning(f"num_sets {num_sets} out of range (1-100), using 1")
                num_sets = 1
        except (TypeError, ValueError):
            logger.warning(f"Invalid num_sets: {num_sets}, using 1")
            num_sets = 1

        if not isinstance(weight_recent, (int, float)) or weight_recent <= 0:
            logger.warning(f"Invalid weight_recent: {weight_recent}, using 1.0")
            weight_recent = 1.0

        logger.info(f"Starting prediction process for {num_sets} sets")

        # Check if we have any data
        if self.df is None or self.df.empty:
            logger.warning("No data available, using statistical prediction")
            return self._statistical_prediction(
                num_sets, weight_recent, favor_hot, favor_due
            )

        # Try to train models if not already trained
        try:
            if not hasattr(self, "regular_model") or not hasattr(self, "strong_model"):
                logger.info("Models not trained, attempting to train...")
                if not self.train_models(weight_recent=weight_recent):
                    logger.warning(
                        "Model training failed, using statistical prediction"
                    )
                    return self._statistical_prediction(
                        num_sets, weight_recent, favor_hot, favor_due
                    )
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return self._statistical_prediction(
                num_sets, weight_recent, favor_hot, favor_due
            )

        # Get statistics for hot/due numbers if needed
        stats = {}
        try:
            if favor_hot or favor_due:
                stats = self.get_number_statistics()
        except Exception as e:
            logger.warning(f"Could not get number statistics: {e}")

        # Generate predictions with error handling
        predictions = []
        max_attempts = num_sets * 2  # Allow some retries for failed predictions

        for attempt in range(max_attempts):
            if len(predictions) >= num_sets:
                break

            try:
                pred_set = self.generate_prediction_set(favor_hot, favor_due, stats)
                if pred_set and self._validate_prediction(pred_set):
                    predictions.append(pred_set)
                    logger.info(
                        f"Generated set {len(predictions)}/{num_sets}: {pred_set['regular_numbers']}, Strong: {pred_set['strong_number']}"
                    )
            except Exception as e:
                logger.error(f"Error generating prediction set: {e}")

        # If we couldn't generate enough predictions, fall back to statistical method
        if len(predictions) < num_sets:
            logger.warning(
                f"Only generated {len(predictions)}/{num_sets} prediction sets, falling back to statistical method"
            )
            remaining = num_sets - len(predictions)
            fallback_preds = self._statistical_prediction(
                remaining, weight_recent, favor_hot, favor_due
            )
            predictions.extend(
                fallback_preds[:remaining]
            )  # Ensure we don't exceed requested number

        # Save predictions
        try:
            if predictions:
                self.save_prediction(predictions)
                logger.info(
                    f"Successfully generated and saved {len(predictions)} prediction sets"
                )
            else:
                logger.error("Failed to generate any predictions")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

        return predictions

    def save_prediction(self, predictions):
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prediction_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sets": len(predictions),
            "prediction_sets": predictions,
            "statistics": self.get_number_statistics(),
        }

        prediction_file = os.path.join(
            self.predictions_dir, f"prediction_{timestamp}.json"
        )

        with open(prediction_file, "w") as f:
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
            f
            for f in os.listdir(self.predictions_dir)
            if f.startswith("prediction_") and f.endswith(".json")
        ]

        if not prediction_files:
            logger.warning("No prediction files available")
            return None

        # Sort by filename (timestamp) to get the latest
        prediction_files.sort(reverse=True)

        latest_file_path = os.path.join(self.predictions_dir, prediction_files[0])

        try:
            with open(latest_file_path, "r") as f:
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

    def predict_next_draw(
        self, num_sets=1, weight_recent=1.0, favor_hot=False, favor_due=False
    ):
        """Predict the next lottery draw"""
        try:
            logger.info(f"Starting prediction process with {num_sets} sets")
            if self.df.empty:
                logger.error("No data available for prediction")
                return []

            predictions = []

            # Try to train models if not already trained
            if not hasattr(self, "regular_model") or not hasattr(self, "strong_model"):
                logger.info("Training models for prediction")
                if not self.train_models():
                    logger.warning(
                        "Model training failed, using statistical prediction"
                    )
                    return self._statistical_prediction(
                        num_sets, weight_recent, favor_hot, favor_due
                    )

            logger.info(f"Generating {num_sets} prediction sets")
            for _ in range(num_sets):
                # Get the last sequence for prediction
                last_sequence = self.df.tail(1).values.flatten()

                # Predict using Random Forest
                rf_pred = self.regular_model.predict([last_sequence])[0]

                # Predict using Neural Network
                # nn_pred = self.nn_model.predict(np.array([last_sequence]), verbose=0)[0]

                # Combine predictions (simple average for now)
                combined_pred = rf_pred

                # Convert predictions to valid lottery numbers
                regular_numbers = self._convert_to_valid_numbers(
                    combined_pred[:6], range(1, 38)
                )
                strong_number = self._convert_to_valid_numbers(
                    combined_pred[6:], range(1, 8)
                )[0]

                predictions.append(
                    {"regular_numbers": regular_numbers, "strong_number": strong_number}
                )

            return predictions
        except Exception as e:
            logger.error(f"Error predicting next draw: {e}")
            logger.warning("Falling back to statistical prediction")
            return self._statistical_prediction(
                num_sets, weight_recent, favor_hot, favor_due
            )


if __name__ == "__main__":
    # Test the predictor
    predictor = LottoPredictor(data_dir="../data")

    # Generate predictions with different strategies
    print("=== Standard Prediction ===")
    standard_pred = predictor.run_prediction_process(num_sets=3)

    print("\n=== Hot Numbers Strategy ===")
    hot_pred = predictor.run_prediction_process(num_sets=3, favor_hot=True)

    print("\n=== Due Numbers Strategy ===")
    due_pred = predictor.run_prediction_process(num_sets=3, favor_due=True)

    print("\n=== Recent Weight Strategy ===")
    recent_pred = predictor.run_prediction_process(num_sets=3, weight_recent=2.0)
