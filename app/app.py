from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="IPL Fantasy Predictor",
    page_icon="🏏",
    layout="wide"
)

st.title("IPL Performance Predictor")
st.markdown("Predict batter runs and bowler wickets using historical IPL data and ML models.")

# Path
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"

BATTER = PROCESSED / "batter_ft.csv"
BOWLER = PROCESSED / "bowler_ft.csv"

BATTER_MODEL = MODELS / "batter_model.pkl"
BOWLER_MODEL = MODELS / "bowler_model.pkl"

# Load data and models
@st.cache_data
def load_data():
    batter = pd.read_csv(BATTER)
    bowler = pd.read_csv(BOWLER)

    batter["date"] = pd.to_datetime(batter["date"], dayfirst=True, errors="coerce")
    bowler["date"] = pd.to_datetime(bowler["date"], dayfirst=True, errors="coerce")

    return batter, bowler

@st.cache_resource
def load_model():
    batter_model = joblib.load(BATTER_MODEL)
    bowler_model = joblib.load(BOWLER_MODEL)

    return batter_model, bowler_model

batter, bowler = load_data()
batter_model, bowler_model = load_model()

# Helper Functions
def latest_batter(df, player, team, opp, venue):
    history = df[df["batter"] == player].sort_values("date")

    if history.empty:
        return None
    
    latest = history.iloc[-1]

    input = pd.DataFrame([{
        "batter" : player,
        "team" : team,
        "venue" : venue,
        "opp" : opp,
        "avg_runs" : latest["avg_runs"],
        "sr" : latest["sr"],
        "avg_balls" : latest["avg_balls"],
        "avg_venue_runs" : latest["avg_venue_runs"],
        "avg_opp_runs" : latest["avg_opp_runs"],
        "avg_4s" : latest["avg_4s"],
        "avg_6s" : latest["avg_6s"],
        "boundary_pct" : latest["boundary_pct"],
        "avg_out" : latest["avg_out"]
    }])
    return input

def latest_bowler(df, player, team, opp, venue):
    history = df[df["bowler"] == player].sort_values("date")

    if history.empty:
        return None
    latest = history.iloc[-1]

    input = pd.DataFrame([{
        "bowler" : player,
        "team" : team,
        "opp" : opp,
        "venue" : venue,
        "avg_wkts" : latest["avg_wkts"],
        "avg_eco" : latest["avg_eco"],
        "avg_runs_given" : latest["avg_runs_given"],
        "avg_venue_wkts" : latest["avg_venue_wkts"],
        "avg_opp_wkts" : latest["avg_opp_wkts"]
    }])
    return input

# Dropdowns
all_teams = sorted(set(batter["team"].dropna().unique()).union(set(bowler["team"].dropna().unique())))
all_opponents = sorted(set(batter["opp"].dropna().unique()).union(set(bowler["opp"].dropna().unique())))
all_venues = sorted(set(batter["venue"].dropna().unique()).union(set(bowler["venue"].dropna().unique())))

def recent_batter(df, team_name):
    team_df = df[df["team"] == team_name].copy()
    if team_df.empty:
        return []
    latest = (
        team_df.sort_values('date', ascending=False).drop_duplicates(subset=["batter"])
    )
    return latest["batter"].dropna().tolist()

def recent_bowler(df, team_name):
    team_df = df[df["team"] == team_name].copy()
    if team_df.empty:
        return []
    latest = (
        team_df.sort_values('date', ascending=False).drop_duplicates(subset=["bowler"])
    )
    return latest["bowler"].dropna().tolist()

# TABS
tab1, tab2 = st.tabs(["Batter Prediction", "Bowler Prediction"])

# BATTER TAB
with tab1:
    st.subheader("Predict Batter Runs")

    col1, col2 = st.columns(2)

    with col1:
        batter_team = st.selectbox("Select Team", all_teams, key="batter_team")

    with col2:
        batter_opp_options = [team for team in all_opponents if team != batter_team]
        batter_opp = st.selectbox("Select Opponents", batter_opp_options, key="batter_opp")

    batter_venue = st.selectbox("Select Venue", all_venues, key="batter_venue")

    avail_batters = recent_batter(batter, batter_team)
    if avail_batters:
        batter_name = st.selectbox("select Batter", avail_batters, key="batter_name")
    else:
        batter_name = None
        st.warning("No batters found for selected team")

    if st.button("Predict Runs", key="predict_runs"):
        if batter_name is not None:
            batter_input = latest_batter(
                batter, batter_name, batter_team, batter_opp, batter_venue
            )
            if batter_input is not None:
                pred_runs = batter_model.predict(batter_input)[0]
                st.success(f"Predicted Runs for {batter_name} : **{pred_runs:.2f}**")
            else:
                st.error("No historical data found for this batter.")

# BOWLER TAB
with tab2:
    st.subheader("Predict Bowler Runs")

    col1, col2 = st.columns(2)

    with col1:
        bowler_team = st.selectbox("Select Team", all_teams, key="bowler_team")

    with col2:
        bowler_opp_options = [team for team in all_opponents if team != bowler_team]
        bowler_opp = st.selectbox("Select Opponents", bowler_opp_options, key="bowler_opp")

    bowler_venue = st.selectbox("Select Venue", all_venues, key="bowler_venue")

    avail_bowlers = recent_bowler(bowler, bowler_team)
    if avail_bowlers:
        bowler_name = st.selectbox("Select Bowler", avail_bowlers, key="bowler_name")
    else:
        batter_name = None
        st.warning("No bowlers found for selected team")

    if st.button("Predict Runs", key="predict_wkts"):
        if bowler_name is not None:
            bowler_input = latest_bowler(
                bowler, bowler_name, bowler_team, bowler_opp, bowler_venue
            )
            if bowler_input is not None:
                pred_wkts = bowler_model.predict(bowler_input)[0]
                st.success(f"Predicted Wicketss for {bowler_name} : **{pred_wkts:.2f}**")
            else:
                st.error("No historical data found for this bowler.")