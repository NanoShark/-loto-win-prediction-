# Loto Win Prediction

This application analyzes historical Israeli Lotto results and provides statistical predictions for upcoming draws.

## Features

- Scrapes historical lottery results from the official Pais website
- Analyzes data to identify patterns and statistics
- Generates predictions for upcoming draws
- Scheduled to run automatically on Sunday and Wednesday mornings
- Web interface to view predictions and statistics

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python src/main.py
   ```

3. Access the web interface at http://localhost:5000

## Data Source

The application uses historical lottery data from the official Pais website:
https://www.pais.co.il/lotto/archive.aspx

## How It Works

The prediction algorithm analyzes historical lottery results using various statistical methods to identify patterns and frequency distributions. It considers factors such as:

- Number frequency
- Hot and cold numbers
- Number pairs and combinations
- Recent trends

The predictions are not guaranteed to win but are based on statistical analysis of historical data.
