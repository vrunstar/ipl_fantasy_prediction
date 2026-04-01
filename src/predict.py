import pandas as pd

batter_df = pd.read_csv("data\processed\batter_ft.csv")
print(batter_df.columns.tolist())

bowler_df = pd.read_csv("data\processed\bowler_stats.csv")
print(bowler_df.columns.tolist())