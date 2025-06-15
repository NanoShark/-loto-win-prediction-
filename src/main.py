#!/usr/bin/env python3
"""
Main script for the Loto Win Prediction application.
This script ties together all components and provides a command-line interface.
"""

import argparse
import logging
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .analyzer import LottoAnalyzer
from .app import app  # Import the Flask app
from .predictor import LottoPredictor
from .scheduler import LottoScheduler
from .scraper import LottoScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/main.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/stats", exist_ok=True)
    os.makedirs("data/predictions", exist_ok=True)
    os.makedirs("data/stats/visualizations", exist_ok=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Loto Win Prediction")

    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape lottery results from the Pais website",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze lottery results and generate statistics",
    )

    parser.add_argument(
        "--predict", action="store_true", help="Generate lottery predictions"
    )

    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Start the scheduler for automatic predictions",
    )

    parser.add_argument("--web", action="store_true", help="Start the web interface")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all components (scrape, analyze, predict, and start web interface)",
    )

    parser.add_argument(
        "--num-sets",
        type=int,
        default=5,
        help="Number of prediction sets to generate (default: 5)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Create necessary directories
    setup_directories()

    # Parse command line arguments
    args = parse_arguments()

    # Initialize components
    data_dir = os.path.abspath("data")
    scraper = LottoScraper(data_dir=data_dir)
    analyzer = LottoAnalyzer(data_dir=data_dir)
    predictor = LottoPredictor(data_dir=data_dir)
    scheduler = LottoScheduler(data_dir=data_dir)

    # Run components based on arguments
    if args.all or args.scrape:
        logger.info("Scraping lottery results")
        scraper.update_results()

    if args.all or args.analyze:
        logger.info("Analyzing lottery results")
        analyzer.generate_statistics()
        analyzer.generate_visualizations()

    if args.all or args.predict:
        logger.info("Generating predictions")
        predictor.run_prediction(num_sets=args.num_sets)

    if args.schedule:
        logger.info("Starting scheduler")
        scheduler.run_scheduler()

    if args.all or args.web:
        logger.info("Starting web interface")
        # Start the scheduler in the background if we're running the web interface
        if not args.schedule:
            scheduler.run_scheduler_in_background()

        # Run the Flask app
        # Binding to '0.0.0.0' makes the application accessible from any network interface.
        # Ensure this is intended and appropriate for the deployment environment.
        # For development, this is common, but for production, consider binding to a specific interface if possible.
        # Debug mode is enabled if FLASK_DEBUG environment variable is set to 'true'
        app.run(
            debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
            host="0.0.0.0",
            port=5000,
        )

    # If no arguments are provided, show help
    if not (
        args.scrape
        or args.analyze
        or args.predict
        or args.schedule
        or args.web
        or args.all
    ):
        logger.info("No arguments provided, showing help")
        print("Loto Win Prediction")
        print("===================")
        print("Usage examples:")
        print("  python main.py --scrape         # Scrape lottery results")
        print("  python main.py --analyze        # Analyze lottery results")
        print("  python main.py --predict        # Generate predictions")
        print("  python main.py --web            # Start web interface")
        print("  python main.py --all            # Run all components")
        print("  python main.py --schedule       # Start scheduler")
        print("\nFor more options, use --help")


if __name__ == "__main__":
    main()
