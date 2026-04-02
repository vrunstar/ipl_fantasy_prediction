from pathlib import Path
import pandas as pd
import joblib

# Path
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

BATTER_FT = PROCESSED / "batter_ft.csv"
BOWLER_FT = PROCESSED / "bowler_ft.csv"

BATTER_MODEL = MODELS / "batter_model.pkl"
BOWLER_MODEL = MODELS / "bowler_model.pkl"


# Load Data
def load_data():
    batter = pd.read_csv(BATTER_FT)
    bowler = pd.read_csv(BOWLER_FT)

    batter["date"] = pd.to_datetime(batter["date"], dayfirst=True, errors="coerce")
    bowler["date"] = pd.to_datetime(bowler["date"],  dayfirst=True, errors="coerce")

    return batter, bowler

def load_model():
    bat_model = joblib.load(BATTER_MODEL)
    bowl_model = joblib.load(BOWLER_MODEL)
    return bat_model, bowl_model

# Latest Batter Features
def latest_batter(batter, bat_name, team, opp, venue):
    player_hist = batter[batter["batter"] == bat_name].sort_values("date")

    if player_hist.empty:
        raise ValueError(f"No history found for batter : {bat_name}")
    latest = player_hist.iloc[-1]
    input = pd.DataFrame([{
        "batter" : bat_name,
        "team" : team,
        "opp" : opp,
        "venue" : venue,
        "avg_runs"  : latest["avg_runs"],
        "sr"  : latest["sr"],
        "avg_balls"  : latest["avg_balls"],
        "avg_venue_runs"  : latest["avg_venue_runs"],
        "avg_opp_runs"  : latest["avg_opp_runs"],
        "avg_4s"  : latest["avg_4s"],
        "avg_6s"  : latest["avg_6s"],
        "boundary_pct"  : latest["boundary_pct"],
        "avg_out"  : latest["avg_out"],
    }])
    return input

# Latest Bowler Features
def latest_bowler(bowler, bowl_name, team, opp, venue):
    player_hist = bowler[bowler["bowler"] == bowl_name].sort_values("date")

    if player_hist.empty:
        raise ValueError(f"No history found for bowler : {bowl_name}")
    latest = player_hist.iloc[-1]
    input = pd.DataFrame([{
        "bowler" : bowl_name,
        "team" : team,
        "opp" : opp,
        "venue" : venue,
        "avg_wkts"  : latest["avg_wkts"],
        "avg_eco"  : latest["avg_eco"],
        "avg_runs_given"  : latest["avg_runs_given"],
        "avg_venue_wkts"  : latest["avg_venue_wkts"],
        "avg_opp_wkts"  : latest["avg_opp_wkts"]
    }])
    return input

# Predictions
def predict_runs(model, batter):
    pred = model.predict(batter)[0]
    return round(pred, 2)

def predict_wkts(model, bowler):
    pred = model.predict(bowler)[0]
    return round(pred, 0)

# Main
def main():
    print("\n>>> Loading data and models ...")
    batter, bowler = load_data()
    bat_model, bowl_model = load_model()

    print("\n====== IPL Performance Predictor ======")
    print("1. Pedict Batter Runs")
    print("2. Pedict Bowler Wkickets")

    choice = int(input("\nEnter choice : "))

    if choice == 1:
        bat_name = input("Enter batter Name : ").strip()
        team = input("Enter teame : ").strip()
        opp = input("Enter opponent : ").strip()
        venue = input("Enter Venue : ").strip()

        try:
            batter_in = latest_batter(
                batter, bat_name, team, opp, venue
            )
            pred_runs = predict_runs(bat_model, batter_in)

            print(f"\nRuns : {pred_runs}")
        except Exception as e:
            print(f"\nError : {e}")
    
    elif choice == 2:
        bowl_name = input("Enter bowler Name : ").strip()
        team = input("Enter team : ").strip()
        opp = input("Enter opponent : ").strip()
        venue = input("Enter Venue : ").strip()

        try:
            bowler_in = latest_bowler(
                bowler, bowl_name, team, opp, venue
            )
            pred_wkts = predict_wkts(bowl_model, bowler_in)

            print(f"\nWkts : {pred_wkts}")
        except Exception as e:
            print(f"\nError : {e}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()