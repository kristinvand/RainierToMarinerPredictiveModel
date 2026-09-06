import pandas as pd
import requests
from datetime import timedelta

CALLUP_FILE = "data/tacoma_callups.csv"
OUTPUT_FILE = "data/callup_model_data.csv"

TEAM_ID = 529
SPORT_ID = 11

FEATURES = [
    "age",
    "plateAppearances",
    "avg",
    "obp",
    "slg",
    "homeRuns",
    "bb_rate",
    "k_rate",
    "stolenBases",
    "babip"
]


def get_tacoma_stats(season, end_date):
    url = "https://statsapi.mlb.com/api/v1/stats"

    params = {
        "stats": "season",
        "group": "hitting",
        "sportIds": SPORT_ID,
        "season": season,
        "teamId": TEAM_ID,
        "gameType": "R",
        "playerPool": "ALL",
        "startDate": f"{season}-03-01",
        "endDate": end_date.strftime("%Y-%m-%d"),
        "limit": 1000
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    splits = data.get("stats", [{}])[0].get("splits", [])

    if not splits:
        return pd.DataFrame()

    return pd.json_normalize(splits)


def prepare_stats(df):
    if df.empty:
        return df

    df["age"] = pd.to_numeric(df["stat.age"], errors="coerce")
    df["plateAppearances"] = pd.to_numeric(
        df["stat.plateAppearances"], errors="coerce"
    )

    df["avg"] = pd.to_numeric(df["stat.avg"], errors="coerce")
    df["obp"] = pd.to_numeric(df["stat.obp"], errors="coerce")
    df["slg"] = pd.to_numeric(df["stat.slg"], errors="coerce")

    df["homeRuns"] = pd.to_numeric(
        df["stat.homeRuns"], errors="coerce"
    )

    df["baseOnBalls"] = pd.to_numeric(
        df["stat.baseOnBalls"], errors="coerce"
    )

    df["strikeOuts"] = pd.to_numeric(
        df["stat.strikeOuts"], errors="coerce"
    )

    df["stolenBases"] = pd.to_numeric(
        df["stat.stolenBases"], errors="coerce"
    )

    df["babip"] = pd.to_numeric(
        df["stat.babip"], errors="coerce"
    )

    df["bb_rate"] = (
        df["baseOnBalls"] / df["plateAppearances"]
    )

    df["k_rate"] = (
        df["strikeOuts"] / df["plateAppearances"]
    )

    return df


callups = pd.read_csv(CALLUP_FILE)
callups["date"] = pd.to_datetime(callups["date"])

# Extract player names from transaction descriptions
callups["player"] = (
    callups["description"]
    .str.extract(r"(?:contract of|recalled) (?:\w+ )?(.+?)(?: from Tacoma|,|$)")
    [0]
    .str.strip()
)

# Keep position players
callups = callups[
    ~callups["description"].str.contains(
        r"\b(?:RHP|LHP)\b",
        case=False,
        na=False
    )
].copy()

callup_events = callups[
    ["date", "player"]
].drop_duplicates()

rows = []

# Use weekly snapshots during each season
for season in range(2021, 2027):
    season_start = pd.Timestamp(f"{season}-04-01")
    season_end = pd.Timestamp(
        "2026-09-06" if season == 2026 else f"{season}-09-30"
    )

    snapshot_date = season_start

    while snapshot_date <= season_end:
        stats = get_tacoma_stats(
            season,
            snapshot_date.to_pydatetime()
        )

        if not stats.empty:
            stats = prepare_stats(stats)

            stats = stats[
                stats["plateAppearances"] >= 25
            ].copy()

            for _, player in stats.iterrows():
                player_name = player["player.fullName"]

                future_callups = callup_events[
                    (callup_events["player"] == player_name)
                    & (callup_events["date"] > snapshot_date)
                    & (
                        callup_events["date"]
                        <= snapshot_date + timedelta(days=7)
                    )
                ]

                called_up = int(len(future_callups) > 0)

                row = {
                    "season": season,
                    "snapshot_date": snapshot_date,
                    "player": player_name,
                    "called_up": called_up
                }

                for feature in FEATURES:
                    row[feature] = player[feature]

                rows.append(row)

        print(
            f"{season} {snapshot_date.strftime('%Y-%m-%d')}: "
            f"{len(stats)} hitters"
        )

        snapshot_date += timedelta(days=7)


model_data = pd.DataFrame(rows)

model_data = model_data.dropna(
    subset=FEATURES + ["called_up"]
)

model_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nRows: {len(model_data)}")
print(
    "\nTarget distribution:"
)
print(
    model_data["called_up"].value_counts()
)

print(f"\nSaved to {OUTPUT_FILE}")