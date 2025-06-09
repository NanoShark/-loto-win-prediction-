from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scraper import LottoScraper
from analyzer import LottoAnalyzer
from predictor import LottoPredictor
from scheduler import LottoScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_numpy_types_to_native(item):
    """Recursively convert NumPy types in a data structure to native Python types."""
    if isinstance(item, dict):
        return {k: convert_numpy_types_to_native(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_numpy_types_to_native(i) for i in item]
    elif isinstance(item, (np.integer, np.int_)):
        return int(item)
    elif isinstance(item, (np.floating, np.float_)):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    return item


app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize components
data_dir = os.path.abspath('data')
scraper = LottoScraper(data_dir=data_dir)
analyzer = LottoAnalyzer(data_dir=data_dir)
predictor = LottoPredictor(data_dir=data_dir)
scheduler = LottoScheduler(data_dir=data_dir)

# Start the scheduler in the background
scheduler_thread = scheduler.run_scheduler_in_background()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predictions')
def predictions():
    """Predictions page"""
    try:
        # Check if we should generate a new prediction based on query parameters
        generate_new = request.args.get('generate', default='false') == 'true'
        
        # Get prediction data
        prediction_data = None
        
        if generate_new:
            # Get parameters from query string with defaults
            num_sets = request.args.get('num_sets', default=6, type=int)
            weight_recent = request.args.get('weight_recent', default=1.0, type=float)
            favor_hot = request.args.get('favor_hot', default='false') == 'true'
            favor_due = request.args.get('favor_due', default='false') == 'true'
            
            # Generate predictions with parameters
            prediction_sets = predictor.run_prediction_process(
                num_sets=num_sets,
                weight_recent=weight_recent,
                favor_hot=favor_hot,
                favor_due=favor_due
            )
            
            # Format prediction data
            prediction_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_sets': prediction_sets
            }
        else:
            # Get the latest prediction
            prediction_data = predictor.get_latest_prediction()
        
        # Get statistics for hot/cold numbers to display on the prediction page
        stats = {}
        try:
            # Try to get prediction stats from analyzer if the method exists
            stats = analyzer.generate_statistics(days=30)
        except Exception as e:
            logger.warning(f"Could not get prediction stats: {e}")
        
        return render_template(
            'predictions.html', 
            prediction_data=prediction_data,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        # Return a template with an error message
        return render_template('predictions.html', error=str(e))

@app.route('/statistics')
def statistics():
    try:
        # Generate statistics
        stats = analyzer.generate_statistics()
        
        # Also get the visualizations for display
        vis_dir = os.path.join(data_dir, 'stats', 'visualizations')
        images = [f for f in os.listdir(vis_dir) if f.endswith('.png')] if os.path.exists(vis_dir) else []
        image_urls = [f"/static/images/{img}" for img in images]
        
        return render_template('statistics.html', stats=stats, image_urls=image_urls)
    except Exception as e:
        logger.error(f"Error in statistics route: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/history')
def history():
    """Lottery history page"""
    # Load the lottery results
    df = scraper.load_results()
    
    if df.empty:
        # If no data exists, scrape it
        df = scraper.scrape_results()
    
    # Convert to list of dictionaries for the template
    results = df.head(100).to_dict('records')
    
    return render_template('history.html', results=results)

@app.route('/api/predictions/latest')
def api_latest_prediction():
    """API endpoint for the latest prediction"""
    try:
        # Get the latest prediction file
        predictions_dir = os.path.join(data_dir, 'predictions')
        prediction_files = [f for f in os.listdir(predictions_dir) if f.startswith('prediction_') and f.endswith('.json')]
        
        if not prediction_files:
            return jsonify({'error': 'No predictions available'}), 404
        
        # Sort by timestamp (newest first)
        prediction_files.sort(reverse=True)
        
        # Load the latest prediction
        latest_file = os.path.join(predictions_dir, prediction_files[0])
        logger.info(f"Loading prediction from {latest_file}")
        
        with open(latest_file, 'r') as f:
            prediction_data = json.load(f)
        
        return jsonify(prediction_data)
    except Exception as e:
        logger.error(f"Error loading prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get parameters from query string or POST data
        if request.method == 'POST':
            # Check if the request has JSON content
            if request.is_json:
                data = request.get_json()
                num_sets = data.get('num_sets', 6)
                weight_recent = float(data.get('weight_recent', 1.0))
                favor_hot = bool(data.get('favor_hot', False))
                favor_due = bool(data.get('favor_due', False))
            else:
                # Handle form data
                num_sets = int(request.form.get('num_sets', 6))
                weight_recent = float(request.form.get('weight_recent', 1.0))
                favor_hot = request.form.get('favor_hot') == 'true'
                favor_due = request.form.get('favor_due') == 'true'
        else:
            # GET method - parameters from query string
            num_sets = request.args.get('num_sets', default=6, type=int)
            weight_recent = request.args.get('weight_recent', default=1.0, type=float)
            favor_hot = request.args.get('favor_hot') == 'true'
            favor_due = request.args.get('favor_due') == 'true'
        
        # Validate parameters
        num_sets = max(1, min(20, num_sets))  # Limit between 1 and 20 sets
        weight_recent = max(1.0, min(5.0, weight_recent))  # Limit between 1.0 and 5.0
        
        logger.info(f"Generating predictions with parameters: num_sets={num_sets}, weight_recent={weight_recent}, favor_hot={favor_hot}, favor_due={favor_due}")
        
        # Run the full prediction process and save results
        prediction_sets = predictor.run_prediction_process(
            num_sets=num_sets,
            weight_recent=weight_recent,
            favor_hot=favor_hot,
            favor_due=favor_due
        )
        
        # Add timestamp and metadata
        response = {
            'success': True,
            'predictions': prediction_sets,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'num_sets': num_sets,
                'weight_recent': weight_recent,
                'favor_hot': favor_hot,
                'favor_due': favor_due
            }
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    try:
        time_range = request.args.get('time_range', 'all')
        days = None if time_range == 'all' else int(time_range)
        logger.info(f"Generating lottery statistics for time range: {time_range} days")
        stats = analyzer.generate_statistics(days=days)
        serializable_stats = convert_numpy_types_to_native(stats)
        return jsonify(serializable_stats)
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def api_history():
    """API endpoint for lottery history"""
    try:
        df = scraper.load_results()
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Get query parameters
        limit = request.args.get('limit', 100, type=int)
        
        # Convert to list of dictionaries
        results = df.head(limit).to_dict('records')
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update-data', methods=['GET', 'POST'])
def update_data():
    try:
        # Update lottery results
        scraper.update_results(force_update=True)
        
        # Regenerate statistics
        analyzer.generate_statistics()
        
        # Generate new visualizations
        analyzer.generate_visualizations()
        
        # Copy visualizations to static folder
        copy_visualizations_to_static()
        
        # If it's an API request (POST), return JSON
        if request.method == 'POST':
            return jsonify({
                'success': True,
                'message': 'Data updated successfully',
                'timestamp': datetime.now().isoformat()
            })
        
        # Otherwise redirect to statistics page
        return redirect(url_for('statistics'))
    except Exception as e:
        app.logger.error(f"Error updating data: {str(e)}")
        
        # If it's an API request (POST), return JSON error
        if request.method == 'POST':
            return jsonify({
                'error': str(e)
            }), 500
            
        # Otherwise render the statistics page with error
        return render_template('statistics.html', error=str(e))

@app.route('/run-prediction', methods=['POST'])
def run_prediction():
    """Run a prediction job immediately"""
    try:
        success = scheduler.run_now()
        
        if success:
            return jsonify({'success': True, 'message': 'Prediction job completed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Prediction job failed'}), 500
    except Exception as e:
        logger.error(f"Error running prediction job: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'stats'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'predictions'), exist_ok=True)
    
    # Create templates and static directories if they don't exist
    templates_dir = os.path.abspath('../templates')
    static_dir = os.path.abspath('../static')
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
