import pandas as pd

# Load the training data
df = pd.read_csv("data/training.csv")

# Calculate plate discipline rates
df["bb_rate"] = df["stat.baseOnBalls"] / df["stat.plateAppearances"]
df["k_rate"] = df["stat.strikeOuts"] / df["stat.plateAppearances"]

# Calculate OPS
df["ops"] = df["stat.obp"] + df["stat.slg"]

# Sort players by season
df = df.sort_values(["player.id", "season"])

# Calculate previous-season statistics
df["previous_ops"] = df.groupby("player.id")["ops"].shift(1)
df["previous_obp"] = df.groupby("player.id")["stat.obp"].shift(1)
df["previous_slg"] = df.groupby("player.id")["stat.slg"].shift(1)
df["previous_bb_rate"] = df.groupby("player.id")["bb_rate"].shift(1)
df["previous_k_rate"] = df.groupby("player.id")["k_rate"].shift(1)

# Calculate year-over-year changes
df["ops_change"] = df["ops"] - df["previous_ops"]
df["obp_change"] = df["stat.obp"] - df["previous_obp"]
df["slg_change"] = df["stat.slg"] - df["previous_slg"]
df["bb_rate_change"] = df["bb_rate"] - df["previous_bb_rate"]
df["k_rate_change"] = df["k_rate"] - df["previous_k_rate"]

# Select modeling features
features = [
    "player.id",
    "player.fullName",
    "season",
    "stat.age",
    "stat.plateAppearances",
    "stat.avg",
    "stat.obp",
    "stat.slg",
    "stat.homeRuns",
    "bb_rate",
    "k_rate",
    "stat.stolenBases",
    "stat.babip",
    "ops_change",
    "obp_change",
    "slg_change",
    "bb_rate_change",
    "k_rate_change",
    "mlb_next_season"
]

model_df = df[features].copy()

# Remove players without enough playing time
model_df = model_df[model_df["stat.plateAppearances"] >= 100]

# Save the modeling dataset
model_df.to_csv("data/model_data.csv", index=False)

# Check the dataset
print(f"Player-seasons: {len(model_df)}")
print(f"Features: {len(features) - 1}")

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nDevelopment data available:")
print(
    f"{model_df['ops_change'].notna().sum()} "
    f"of {len(model_df)} player-seasons"
)

print("\nSaved to data/model_data.csv")