import logging
import threading
import time

import schedule

from .analyzer import LottoAnalyzer
from .predictor import LottoPredictor
from .scraper import LottoScraper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/scheduler.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LottoScheduler:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.scraper = LottoScraper(data_dir=data_dir)
        self.analyzer = LottoAnalyzer(data_dir=data_dir)
        self.predictor = LottoPredictor(data_dir=data_dir)

    def run_prediction_job(self):
        """
        Run the full prediction job:
        1. Update lottery results
        2. Generate statistics
        3. Generate predictions
        """
        logger.info("Starting scheduled prediction job")

        try:
            # Update lottery results
            logger.info("Updating lottery results")
            self.scraper.update_results()

            # Generate statistics
            logger.info("Generating statistics")
            self.analyzer.generate_statistics()
            self.analyzer.generate_visualizations()

            # Generate predictions
            logger.info("Generating predictions")
            prediction = self.predictor.run_prediction(num_sets=5)

            logger.info("Prediction job completed successfully")

            # Log the predictions
            logger.info("Prediction sets:")
            for i, pred_set in enumerate(prediction, 1):
                logger.info(
                    f"Set {i}: Regular numbers: {pred_set['regular_numbers']}, Strong number: {pred_set['strong_number']}"
                )

            return True
        except Exception as e:
            logger.error(f"Error in prediction job: {e}")
            return False

    def schedule_jobs(self):
        """
        Schedule prediction jobs for Sunday and Wednesday mornings
        """
        # Schedule for Sunday morning at 8:00 AM
        schedule.every().sunday.at("08:00").do(self.run_prediction_job)

        # Schedule for Wednesday morning at 8:00 AM
        schedule.every().wednesday.at("08:00").do(self.run_prediction_job)

        logger.info("Scheduled prediction jobs for Sunday and Wednesday at 8:00 AM")

    def run_scheduler(self):
        """
        Run the scheduler in a loop
        """
        self.schedule_jobs()

        logger.info("Scheduler started")

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_scheduler_in_background(self):
        """
        Run the scheduler in a background thread
        """
        thread = threading.Thread(target=self.run_scheduler)
        thread.daemon = True
        thread.start()

        logger.info("Scheduler started in background thread")

        return thread

    def run_now(self):
        """
        Run a prediction job immediately
        """
        logger.info("Running prediction job immediately")
        return self.run_prediction_job()


if __name__ == "__main__":
    # Test the scheduler
    scheduler = LottoScheduler(data_dir="../data")

    # Run a prediction job immediately
    scheduler.run_now()

    # Schedule future jobs
    scheduler.schedule_jobs()

    print("Scheduler test complete. Press Ctrl+C to exit.")

    try:
        # Keep the script running
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Scheduler stopped.")
