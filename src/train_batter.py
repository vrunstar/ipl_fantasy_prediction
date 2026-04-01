from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Paths -----------------------------------
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

BATTER = PROCESSED / "batter_ft.csv"
MODEL = MODELS / "batter_model.pkl"

MODELS.mkdir(parents=True, exist_ok=True)

# Loading Data -----------------------------------
def load():
    df = pd.read_csv(BATTER)
    df["date"] = pd.to_datetime(df["date"])
    return df

# Prepare Data -----------------------------------
def prepare(df):
    feet = [
        "batter", 'team', 'opp', 'venue', 'avg_runs', 'sr', 'avg_balls', 'avg_venue_runs',
        'avg_opp_runs', 'avg_4s', 'avg_6s', 'boundary_pct', 'avg_out'
    ]

    target = "runs"

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    x = df[feet]
    y = df[target]

    return df, x, y, feet

# Pipeline -----------------------------------
def pipeline(cate, nums):
    cate_trans = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    nums_trans = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocess = ColumnTransformer(transformers=[
        ("cat", cate_trans, cate),
        ("num", nums_trans, nums)  
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=69,
        n_jobs=1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    return pipe

# Time Split -----------------------------------
def time_split(x, y, train_ratio=0.8):
    split_idx = int(len(x) * train_ratio)

    x_train = x.iloc[:split_idx]
    x_test = x.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return x_train, x_test, y_train, y_test

# train & evaluate Model -----------------------------------
def train_eval(x, y):
    cate = ['batter', 'team', 'opp', 'venue']
    nums = [col for col in x.columns if col not in cate]

    x_train, x_test, y_train, y_test = time_split(x, y, train_ratio=0.8)

    pipe = pipeline(cate, nums)

    print("\n>>> Training Model ...")
    pipe.fit(x_train, y_train)

    print(">>> Predicting ...")
    y_pred = pipe.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n>>> Batter Runs Model Prformance")
    print("-"*35)
    print(f"Train Size : {len(x_train)}")
    print(f"test Size  : {len(x_test)}")
    print("-"*35)
    print(f"MAE        : {mae:.2f}")
    print(f"RMSE       : {rmse:.2f}")
    print(f"R² Score   : {r2:.4f}")

    result = pd.DataFrame({
        "Actual Runs" : y_test.values,
        "Predicted Runs" : y_pred
    })
    result["Error"] = result["Actual Runs"] - result["Predicted Runs"]

    return pipe, result

# Save Model -----------------------------------
def save(model):
    joblib.dump(model, MODEL)
    print(f"\nModel saved to : {MODEL}")

# Main -----------------------------------
def main():
    print("]n>>> Loading dataset ...")
    df = load()
    print(f">>> Dataset shape : {df.shape}")

    print("\n>>> Preparing Data ...")
    df, x, y, feet = prepare(df)
    print(f">>> Features used : \n{feet}")

    model, result = train_eval(x, y)

    save(model)

    print(f"\n>>> Sample Predictions")
    print("-" * 100)
    print(result.head(10))
    print("-" * 100)

    print("\n>>> Done!")

if __name__ == "__main__":
    main()