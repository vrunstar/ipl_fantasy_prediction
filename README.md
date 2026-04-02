# IPL Fantasy Predictor

An end-to-end machine learning project for analyzing **Indian Premier League (IPL)** cricket data and predicting **batter runs** and **bowler wickets** using historical player performance and match context.

This project combines:
- cricket data analysis
- fantasy cricket logic
- feature engineering
- machine learning model training
- an interactive Streamlit app

---

## Project Overview

Player performance in T20 cricket depends on multiple contextual factors such as:

- recent form
- opposition strength
- venue conditions
- batting / bowling consistency
- scoring and wicket-taking patterns

This project was built to estimate player performance in IPL matches using historical data and machine learning.

### Predictions supported:
- **Batter Runs Prediction**
- **Bowler Wickets Prediction**

It can also serve as a foundation for **fantasy cricket analysis** and future fantasy-point forecasting.

---

## Features

- **IPL Data Analysis**  
  Explore batting and bowling performance trends from historical IPL match data.

- **Fantasy Cricket Logic**  
  Supports fantasy-oriented cricket analytics and can be extended to estimate player fantasy points.

- **Machine Learning Models**  
  Separate predictive models for:
  - batter runs
  - bowler wickets

- **Interactive Dashboard**  
  Streamlit-based app for selecting teams, players, opponents, and venues to generate predictions.

- **Jupyter Notebooks**  
  Notebooks for:
  - cleaning and EDA
  - feature engineering
  - model experimentation
  - fantasy logic exploration

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure

```bash
ipl/
├── app/
│   └── app.py                   # Streamlit web application
├── data/
│   ├── raw/
│   │   └── .gitkeep             # Raw dataset folder (not tracked)
│   └── processed/
│       ├── batter_ft.csv        # Processed batter features
│       └── bowler_ft.csv        # Processed bowler features
├── models/
│   ├── batter_runs_model.pkl    # Trained batter prediction model
│   ├── bowler_wickets_model.pkl # Trained bowler prediction model
│   └── .gitkeep
├── notebooks/
│   ├── batter_model.ipynb       # Batter model experimentation
│   ├── bowler_model.ipynb       # Bowler model experimentation
│   ├── cleaning_eda.ipynb       # Data cleaning and EDA
│   ├── fantasy_points.ipynb     # Fantasy logic exploration
│   └── features.ipynb           # Feature engineering
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── explain.py               # Model explanation / helper utilities
│   ├── fantasy.py               # Fantasy points logic
│   ├── features.py              # Feature engineering
│   ├── predict.py               # Prediction script
│   ├── preprocess.py            # Data preprocessing
│   ├── train_batter.py          # Batter model training
│   ├── train_bowler.py          # Bowler model training
│   └── utils.py                 # Utility functions
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files/folders
└── README.md                    # Project documentation
```

---

## Dataset Note
- This repository includes processed IPL player-level datasets used for training and prediction.
- The original raw IPL ball-by-ball dataset is not included in the repository to keep the project lightweight and easier to run.
- 
---

## Machine Learning Workflow

1. Data Preparation
- loaded and cleaned IPL player-level data
- separated batter and bowler performance records
- prepared structured datasets for downstream modeling

2. Feature Engineering
- Built contextual features such as:
   - Batter Features
      - historical average runs
      - strike rate
      - average balls faced
      - average 4s
      - average 6s
      - boundary percentage
      - venue average runs
      - opponent average runs
      - dismissal / out tendency
   - Bowler Features
      - historical average wickets
      - economy rate      
      - average runs conceded      
      - venue average wickets      
      - opponent average wickets

3) Model Training
- Trained separate machine learning models for\
   - Batter prediction → target: runs
   - Bowler prediction → target: wkts
  
4) Model Evaluation\
- Models were evaluated using:
   - MAE
   - RMSE
   - R² Score
---
## Streamlit App
- The project includes an interactive Streamlit app that allows users to:
   - select a team  
   - select an opponent
   - select a venue
- choose recent/current players from that team
- generate batter or bowler predictions instantly

---

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd ipl
```
2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the Streamlit App
```bash
streamlit run app/app.py
```
Run Prediction Script
```bash
python src/predict.py
```
Train Models
```bash
python src/train_batter.py
python src/train_bowler.py
```
---

## Future Improvements
- better current-squad filtering
- fantasy points prediction
- richer player stat cards in app
- model explainability visuals
- advanced models like:
   - XGBoost   
   - LightGBM   
   - CatBoost
---

## Author

Varun Shakya \
Machine Learning Student
