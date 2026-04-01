from pathlib import Path
import pandas as pd

# Path ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR/"data"/"processed"

BATTER = PROCESSED/"batter_stats.csv"
BOWLER = PROCESSED/"bowler_stats.csv"

BATTER_FT = PROCESSED/"batter_ft.csv"
BOWLER_FT = PROCESSED/"bowler_ft.csv"


# Loading ------------------
def load():
    batter = pd.read_csv(BATTER)
    bowler = pd.read_csv(BOWLER)

    batter["date"] = pd.to_datetime(batter["date"])
    bowler["date"] = pd.to_datetime(bowler["date"])

    return batter, bowler

# Batter Features ------------------
def batter_feet(df):
    df = df.copy()
    df = df.sort_values(["batter", "date", "match_id"]).reset_index(drop=True)

    df['avg_runs'] = (
        df.groupby("batter")["runs"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    
    df['sr'] = (
        df.groupby("batter")['SR']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['avg_balls'] = (
        df.groupby("batter")['balls']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['avg_venue_runs'] = (
        df.groupby(["batter", "venue"])['runs']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['avg_opp_runs'] = (
        df.groupby(["batter", "opp"])['runs']
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    df['avg_4s'] = (
        df.groupby("batter")['fours']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['avg_6s'] = (
        df.groupby("batter")['sixes']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['boundary_pct'] = (
        df.groupby("batter")
        .apply(lambda g: ((g["fours"] + g["sixes"]) / g["balls"].replace(0, 1)).shift(1).rolling(5, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df["avg_out"] = (
        df.groupby("batter")['outs']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    return df

# Bowler Features ------------------
def bowler_feet(df):
    df = df.copy()
    df = df.sort_values(["bowler", "date", "match_id"]).reset_index(drop=True)

    df['avg_wkts'] = (
        df.groupby("bowler")["wkts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['avg_eco'] = (
        df.groupby("bowler")['econ']
        .transform(lambda x: x.shift(1).rolling(5,min_periods=1).mean())
    )

    df['avg_runs_given'] = (
        df.groupby("bowler")['runs']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    df['avg_venue_wkts'] = (
        df.groupby(["bowler", "venue"])['wkts']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['avg_opp_wkts'] = (
        df.groupby(["bowler", "opp"])['wkts']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    return df

# Missing Values ------------------
def missing(df, feet):
    df = df.copy()
    for col in feet:
        df[col] = df[col].fillna(0)
    return df

# Saving ------------------
def save(bat, bowl):
    bat.to_csv(BATTER_FT, index=False)
    bowl.to_csv(BOWLER_FT, index=False)

    print(f"Saved batter features at : {BATTER_FT}")
    print(f"Saved bowler features at : {BOWLER_FT}")

def main():
    print("\n>>> Loading processed stats ...")
    batter, bowler = load()

    print("\n>>> Creating batter features ...")
    batter_ft = batter_feet(batter)
    
    batter_ft_cols = ['avg_runs', 'sr', 'avg_balls', 'avg_balls', 'avg_venue_runs', 'avg_opp_runs', 'avg_4s', 'avg_6s', 'boundary_pct', 'avg_out']
    batter_ft = missing(batter_ft, batter_ft_cols)
    
    print(f"\n>>> Batter features shape : {batter_ft.shape}")
    print(f">>> Preview\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n{batter_ft.head()}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    
    print("\n>>> Creating batter features ...")
    bowler_ft = bowler_feet(bowler)
    
    bowler_ft_cols = ['avg_wkts', 'avg_eco', 'avg_runs_given', 'avg_venue_wkts', 'avg_opp_wkts']
    bowler_ft = missing(bowler_ft, bowler_ft_cols)
    
    print(f"\n>>> Bowler features shape : {bowler_ft.shape}")
    print(f">>> Preview\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n{bowler_ft.head()}")
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------------")
    
    print("\n>>> Saving feature datasets ... ")
    save(batter_ft, bowler_ft)

    print("\n>>> Done!")

if __name__ == "__main__":
    main()