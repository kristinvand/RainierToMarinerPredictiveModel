import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "data/callup_model_data.csv"
CALLUPS_FILE = "data/tacoma_callups.csv"
OUTPUT_FILE = "data/2026_callup_predictions.csv"
FIGURE_FILE = "visualizations/2026_callup_predictions.png"

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

TARGET = "called_up"


data = pd.read_csv(DATA_FILE)
data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

callups = pd.read_csv(CALLUPS_FILE)
callups["date"] = pd.to_datetime(callups["date"])


# Train on 2021-2025 and hold out 2026
train = data[data["season"] <= 2025].copy()
test = data[data["season"] == 2026].copy()

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print(f"Training observations: {len(train)}")
print(f"2026 observations:     {len(test)}")
print(f"Training call-ups:     {y_train.sum()}")
print(f"2026 call-ups:         {y_test.sum()}")


# Logistic regression
logistic = Pipeline([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

logistic.fit(X_train, y_train)

logistic_prob = logistic.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("-------------------")
print(f"ROC-AUC: {roc_auc_score(y_test, logistic_prob):.3f}")


# Random forest
random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

random_forest.fit(X_train, y_train)

rf_prob = random_forest.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print("-------------")
print(f"ROC-AUC: {roc_auc_score(y_test, rf_prob):.3f}")


# Ranking performance
def precision_at_k(probabilities, actual, k):
    ranked = pd.DataFrame({
        "score": probabilities,
        "actual": actual.values
    }).sort_values("score", ascending=False)

    return ranked.head(k)["actual"].mean()


print("\nPrecision@K")
print("-----------")

for k in [5, 10, 20]:
    print(
        f"Top {k}: "
        f"{precision_at_k(logistic_prob, y_test, k):.3f} Logistic | "
        f"{precision_at_k(rf_prob, y_test, k):.3f} Random Forest"
    )


# Random forest performed better on the 2026 holdout
model = random_forest


# Use the latest available 2026 snapshot
current = (
    test
    .sort_values("snapshot_date")
    .groupby("player", as_index=False)
    .tail(1)
    .copy()
)


# Determine which players were still in Seattle at the cutoff
cutoff = pd.Timestamp("2026-09-06")

transactions = pd.read_csv("data/mariners_transactions.csv")
transactions["date"] = pd.to_datetime(transactions["date"])

transactions = transactions[
    transactions["date"] < cutoff
].copy()

transactions["player"] = (
    transactions["person.fullName"]
    .fillna("")
    .str.strip()
)

tacoma_transactions = transactions[
    transactions["description"].str.contains(
        "Tacoma Rainiers",
        case=False,
        na=False
    )
].copy()


# Find each player's latest Tacoma-related transaction
latest_transaction = (
    tacoma_transactions
    .sort_values("date")
    .groupby("player", as_index=False)
    .tail(1)
)


# Players whose latest Tacoma transaction moved them to Seattle
in_seattle = set(
    latest_transaction.loc[
        latest_transaction["description"].str.contains(
            "from Tacoma Rainiers",
            case=False,
            na=False
        ),
        "player"
    ]
)


# Keep players currently eligible for another call-up
current = current[
    ~current["player"].isin(in_seattle)
].copy()


# Generate model scores
current["model_score"] = model.predict_proba(
    current[FEATURES]
)[:, 1]


predictions = current[
    [
        "player",
        "snapshot_date",
        "age",
        "plateAppearances",
        "avg",
        "obp",
        "slg",
        "homeRuns",
        "bb_rate",
        "k_rate",
        "stolenBases",
        "babip",
        "model_score"
    ]
].sort_values(
    "model_score",
    ascending=False
)


predictions.to_csv(OUTPUT_FILE, index=False)


print("\nPredicted next Tacoma call-ups")
print("------------------------------")

print(
    predictions[
        ["player", "model_score"]
    ].head(10).to_string(index=False)
)

print(f"\nEligible players: {len(predictions)}")
print(f"Saved to {OUTPUT_FILE}")


# Create prediction visualization
top = predictions.head(10).sort_values("model_score")

plt.figure(figsize=(9, 6))

plt.barh(
    top["player"],
    top["model_score"]
)

plt.xlabel("Model Score")
plt.ylabel("")
plt.title("Who's Next? — Tacoma to Seattle Call-Up Model")
plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    FIGURE_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved visualization to {FIGURE_FILE}")