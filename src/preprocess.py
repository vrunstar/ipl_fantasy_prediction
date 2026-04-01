from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

DATA_PATH = RAW_DATA_DIR / "ipl.csv"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df

def standardize_cols(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def batter_stats(df):
    batter = df.copy()

    batter_stats = batter.groupby(["match_id", "date", "venue","innings", "batting_team", "bowling_team", "batter"], as_index=False).agg(
        runs = ("runs_batter", "sum"),
        balls = ("balls_faced", "count"),
        fours = ("runs_batter", lambda x: (x==4).sum()),
        sixes = ("runs_batter", lambda x: (x==6).sum()),
        team_score = ("team_runs", "min"),
        outs = ("player_out", "count")
    )
    
    batter_stats["SR"] = ((batter_stats["runs"] / batter_stats["balls"].replace(0,1))*100).round(2)

    batter_stats.rename(columns={
        'batting_team':'team',
        'bowling_team' : 'opp',
    }, inplace=True)

    return batter_stats

def bowler_stats(df):
    bowler = df.copy()

    bowler_stats = bowler.groupby(["match_id", "date", "venue","innings", "bowling_team", "batting_team", "bowler"], as_index=False).agg(
        balls = ("ball", "count"),
        runs = ("runs_bowler", "sum"),
        wkts = ("bowler_wicket", "sum")
        )

    bowler_stats['overs'] = (bowler_stats['balls']/6).round(2)
    bowler_stats['econ'] = (bowler_stats['runs'] / bowler_stats['overs'].replace(0,1)).round(2)

    bowler_stats.rename(columns={
        'batting_team':'team',
        'bowling_team' : 'opp',
    }, inplace=True)

    return bowler_stats

def save_processed(batter_stats, bowler_stats):
    batter_path = PROCESSED_DATA_DIR / "batter_stats.csv"
    bowler_path = PROCESSED_DATA_DIR / "bowler_stats.csv"

    batter_stats.to_csv(batter_path, index=False)
    bowler_stats.to_csv(bowler_path, index=False)

    print(f"Saved Batter Stats to : {batter_path}")
    print(f"Saved Bowler Stats to : {bowler_path}")

def main():
    print("\n\n>>> Loading raw IPL data ...")
    df = load_raw_data()

    print("\n>>> Standardizing Columns ...")
    df = standardize_cols(df)

    print(f">>> Dataset Shape : {df.shape}")

    print("\n>>> Creating Batter dataset ...")
    batter = batter_stats(df)
    print(f">>> Batter dataset shape : {batter.shape}")
    print(f">>> Preview\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n{batter.head()}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("\n>>> Creating Bowler dataset ...")
    bowler = bowler_stats(df)
    print(f">>> Bowler dataset shape : {bowler.shape}")
    print(f">>> Preview\n----------------------------------------------------------------------------------------------------------------------------------------------------------------\n{bowler.head()}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("\n>>> Saving processed files ...")
    save_processed(batter, bowler)

    print("\n>>> Done!")

if __name__ == "__main__":
    main()