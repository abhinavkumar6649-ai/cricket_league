# 🏏 Cricket League Score Prediction

An end-to-end Machine Learning project to predict cricket match scores based on match conditions, team fitness, toss results, venue advantage, opposition bowling strength, and historical team performance.

---

## 📌 Project Overview

In competitive cricket leagues (such as inter-college tournaments), match-day conditions and team metrics heavily influence a team's total runs scored. This project implements a comprehensive data science and machine learning workflow:
1. **Synthetic Data Simulation**: Simulates realistic match data across engineering branches (CSE, ECE, Mechanical, Civil, EEE) with injected real-world data quality issues (missing values, corrupted negative runs).
2. **Exploratory Data Analysis (EDA)**: Visualizations exploring relationships between toss outcome, match importance (regular vs. final), opposition bowling quality, and runs scored.
3. **Data Cleaning & Feature Engineering**: Imputing missing fitness indices, correcting anomalies, generating date-based features (day of week, month), calculating expanding rolling average runs per branch, and one-hot encoding categorical variables.
4. **Machine Learning Modeling**: Training and evaluating multiple regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
5. **Hyperparameter Tuning**: Optimizing Random Forest hyperparameters using `RandomizedSearchCV`.
6. **Inference & Prediction**: Predicting match scores for hypothetical game scenarios.

---

## 📊 Dataset Description

The dataset simulates 200 matches with the following attributes:

| Feature | Type | Description |
|---|---|---|
| `date` | Date | Match date |
| `branch` | Categorical | Batting team branch (`CSE`, `ECE`, `Mechanical`, `Civil`, `EEE`) |
| `is_toss_won` | Boolean | Whether the batting team won the toss |
| `is_home_ground` | Boolean | Whether the match is played at home ground |
| `is_final` | Boolean | Whether the match is a championship final |
| `team_fitness_index` | Float | Team physical fitness rating (~50–100) |
| `opposition_bowling_rating` | Float | Strength of opposing bowling attack (~40–100) |
| `rolling_avg_runs` | Float | Historical expanding average runs scored by branch |
| `runs_scored` | Integer | **Target variable**: Total runs scored by the team |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/abhinavkumar6649-ai/cricket_league.git
cd cricket_league
pip install -r requirements.txt
```

### 3. Running the Notebook
Launch Jupyter Notebook:

```bash
jupyter notebook cricket_league.ipynb
```
Run all cells sequentially. The notebook will automatically generate `cricket_league_data.csv`, preprocess the data, evaluate models, and display evaluation metrics and visualizations.

---

## 📈 Model Performance

| Model | Test RMSE | Test MAE | Test $R^2$ Score |
|---|---|---|---|
| **Linear Regression** | ~9.63 | ~7.78 | ~0.76 |
| **Decision Tree** | ~14.80 | ~11.83 | ~0.43 |
| **Random Forest Regressor** | ~9.79 | ~7.85 | ~0.75 |

---

## 🔮 Sample Inference

Predicting total runs for a future match:
- **Branch**: CSE
- **Toss**: Won (`True`)
- **Venue**: Home Ground (`True`)
- **Match Stage**: Regular Match (`False`)
- **Fitness Index**: `82`
- **Opposition Bowling Rating**: `68`
- **Rolling Average Runs**: `150`

```python
# Predicted Runs: ~170 - 173 runs
```

---

## 📁 Repository Structure

```
cricket_league/
├── cricket_league.ipynb       # Main Jupyter Notebook with full ML pipeline
├── cricket_league_data.csv    # Generated simulation dataset
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```
