# IPL Data Analysis and Fantasy Cricket Predictor

This project provides a comprehensive analysis of Indian Premier League (IPL) cricket data, including player statistics, fantasy points calculation, and machine learning models for predicting batter and bowler performance.

## Features

- **Data Analysis**: Explore IPL match data with detailed statistics for batters and bowlers
- **Fantasy Points Calculator**: Calculate fantasy cricket points based on player performance
- **Machine Learning Models**: Predict player performance using trained models for batters and bowlers
- **Interactive Dashboard**: Streamlit web application for visualizing data and predictions
- **Jupyter Notebooks**: Detailed analysis and model development notebooks

## Project Structure

```
ipl/
├── app/
│   └── app.py                    # Main Streamlit application
├── data/
│   ├── raw/
│   │   └── IPL.csv              # Raw IPL match data
│   └── processed/
│       ├── batter_ft.csv        # Processed batter fantasy data
│       ├── batter_stats.csv     # Processed batter statistics
│       ├── bowler_ft.csv        # Processed bowler fantasy data
│       └── bowler_stats.csv     # Processed bowler statistics
├── models/                      # Trained machine learning models
├── notebooks/
│   ├── batter_model.ipynb       # Batter performance model development
│   ├── bowler_model.ipynb       # Bowler performance model development
│   ├── cleaning_eda.ipynb       # Data cleaning and exploratory analysis
│   ├── fantasy_points.ipynb     # Fantasy points calculation
│   ├── features.ipynb           # Feature engineering
│   └── test.ipynb               # Testing and validation
├── output/
│   └── charts/                  # Generated charts and visualizations
├── outputs/                     # Additional output files
├── tablets/                     # Tablet-specific outputs
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── explain.py               # Model explanation tools
│   ├── fantasy.py               # Fantasy points calculation
│   ├── features.py              # Feature engineering
│   ├── predict.py               # Prediction functions
│   ├── preprocess.py            # Data preprocessing
│   ├── train_batter.py          # Batter model training
│   ├── train_bowler.py          # Bowler model training
│   └── utils.py                 # Utility functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ipl
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

To launch the interactive dashboard:

```bash
streamlit run app/app.py
```

This will start a web server where you can explore IPL data, view player statistics, and make predictions.

### Data Processing

The project includes scripts for data processing:

- `src/data_loader.py`: Load and process raw IPL data
- `src/preprocess.py`: Clean and prepare data for analysis
- `src/features.py`: Generate features for machine learning models

### Model Training

Train the machine learning models:

- Batter model: `python src/train_batter.py`
- Bowler model: `python src/train_bowler.py`

### Making Predictions

Use the prediction module:

```python
from src.predict import predict_batter_performance, predict_bowler_performance

# Example usage
batter_pred = predict_batter_performance(player_data)
bowler_pred = predict_bowler_performance(player_data)
```

## Deployment

### Streamlit Cloud

To deploy your app on Streamlit Cloud:

1. Ensure your code is pushed to a public GitHub repository.

2. Go to [share.streamlit.io](https://share.streamlit.io).

3. Connect your GitHub account and select your repository.

4. Set the main file path to `app/app.py`.

5. Add any advanced settings if needed (e.g., Python version).

6. Click "Deploy".

**Notes:**
- Make sure all data files in `data/` are committed to the repository, as Streamlit Cloud will access them.
- If your app uses large datasets, consider optimizing data loading or using Streamlit's caching.
- For private repositories, you need a paid plan or make it public temporarily for deployment.

### Local Deployment

For local deployment or other platforms, ensure the virtual environment is activated and run:

```bash
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

## Data

The project uses IPL match data stored in `data/raw/IPL.csv`. Processed data includes:

- Player statistics (runs, wickets, etc.)
- Fantasy points calculations
- Feature-engineered data for modeling

## Models

Machine learning models are trained to predict:

- Batter performance metrics (runs, strike rate, etc.)
- Bowler performance metrics (wickets, economy, etc.)

Models are saved in the `models/` directory and can be loaded for inference.

## Notebooks

Explore the analysis and model development through Jupyter notebooks:

- `cleaning_eda.ipynb`: Data cleaning and exploratory data analysis
- `fantasy_points.ipynb`: Fantasy cricket points system
- `features.ipynb`: Feature engineering process
- `batter_model.ipynb`: Batter model development and evaluation
- `bowler_model.ipynb`: Bowler model development and evaluation
- `test.ipynb`: Model testing and validation

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.