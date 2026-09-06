import pandas as pd
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the historical modeling data
df = pd.read_csv("data/model_data.csv")

# Select model features
features = [
    "stat.age",
    "stat.plateAppearances",
    "stat.avg",
    "stat.obp",
    "stat.slg",
    "stat.homeRuns",
    "bb_rate",
    "k_rate",
    "stat.stolenBases",
    "stat.babip"
]

# Train on completed seasons
train = df[df["season"].isin([2021, 2022, 2023, 2024])]

X_train = train[features]
y_train = train["mlb_next_season"]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# MLB Stats API endpoint
url = "https://statsapi.mlb.com/api/v1/teams"

# Find Tacoma's 2026 team ID
params = {
    "sportId": 11,
    "season": 2026
}

response = requests.get(url, params=params)
response.raise_for_status()

teams = response.json()["teams"]

tacoma = next(
    team for team in teams
    if team["name"] == "Tacoma Rainiers"
)

team_id = tacoma["id"]

print(f"Tacoma team ID: {team_id}")

# Get Tacoma's 2026 hitting stats
url = "https://statsapi.mlb.com/api/v1/stats"

params = {
    "stats": "season",
    "group": "hitting",
    "sportIds": 11,
    "season": 2026,
    "teamId": team_id,
    "gameType": "R",
    "playerPool": "ALL",
    "limit": 1000,
    "offset": 0
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# Convert player stats to a dataframe
splits = data["stats"][0]["splits"]

rainiers = pd.json_normalize(splits)

# Calculate model features
rainiers["bb_rate"] = (
    rainiers["stat.baseOnBalls"] /
    rainiers["stat.plateAppearances"]
)

rainiers["k_rate"] = (
    rainiers["stat.strikeOuts"] /
    rainiers["stat.plateAppearances"]
)

# Keep players with enough playing time
rainiers = rainiers[
    rainiers["stat.plateAppearances"] >= 100
].copy()

# Select model features
X_rainiers = rainiers[features]

# Standardize using the training scaler
X_rainiers_scaled = scaler.transform(X_rainiers)

# Predict MLB probability
rainiers["mlb_probability"] = (
    model.predict_proba(X_rainiers_scaled)[:, 1]
)

# Add player names
rainiers["player_name"] = rainiers["player.fullName"]

# Sort by predicted probability
predictions = rainiers.sort_values(
    "mlb_probability",
    ascending=False
)

# Select columns for the final table
output = predictions[
    [
        "player_name",
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
        "mlb_probability"
    ]
].copy()

# Convert probability to percentage
output["mlb_probability"] = (
    output["mlb_probability"] * 100
).round(1)

print("\n2026 Tacoma Rainiers MLB Predictions")
print(
    output.to_string(
        index=False
    )
)

# Save predictions
output.to_csv(
    "data/2026_rainiers_predictions.csv",
    index=False
)

print("\nSaved to data/2026_rainiers_predictions.csv")