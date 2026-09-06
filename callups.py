import pandas as pd


INPUT_FILE = "data/mariners_transactions.csv"
OUTPUT_FILE = "data/tacoma_callups.csv"


transactions = pd.read_csv(INPUT_FILE)

callups = transactions[
    transactions["description"].str.contains(
        "from Tacoma Rainiers",
        case=False,
        na=False
    )
    & transactions["typeDesc"].isin(["Selected", "Recalled"])
].copy()

if "person.fullName" not in callups.columns:
    raise ValueError("person.fullName was not found in the transaction data.")

callups["player"] = callups["person.fullName"].str.strip()

callups = callups[
    ["date", "player", "typeDesc", "description"]
].sort_values("date")

callups.to_csv(OUTPUT_FILE, index=False)

print(f"Total Tacoma-to-Seattle call-ups: {len(callups)}")
print("\nCall-ups by year:")
print(pd.to_datetime(callups["date"]).dt.year.value_counts().sort_index())

print(f"\nSaved to {OUTPUT_FILE}")