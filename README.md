# Rainier to Mariner

## Who's Next?

Can minor league performance help predict which Tacoma Rainiers player is next to receive a Seattle Mariners call-up?

This project uses historical Tacoma-to-Seattle transactions and player performance data to build a machine learning model that ranks current Rainiers hitters by their likelihood of being the next call-up.

### 2026 Model

| Model | 2026 ROC-AUC |
|---|---:|
| Random Forest | **0.701** |
| Logistic Regression | 0.660 |

The Random Forest performed best on a chronological 2026 holdout and was used to generate the final player rankings.

### What I Found

The model ranked **Michael Arroyo #3 among eligible Tacoma hitters** immediately before his September 6, 2026 promotion to Seattle.

This is a proof-of-concept rather than a claim that the model can predict front-office decisions. Call-ups depend on factors beyond player performance, including roster availability, injuries, positional needs, and organizational decisions.

![2026 Tacoma Call-Up Predictions](visualizations/2026_callup_predictions.png)

---

## Project Overview

The project began with a broader question:

> Can a player's Triple-A performance be used to predict whether they will reach MLB in the following season?

I built a supervised learning dataset using Triple-A hitters from 2021–2024 and whether they appeared in MLB the following season. I first established a baseline model using current-season performance, then tested whether year-over-year development features improved predictions.

Those development features did not consistently improve performance.

That led to a more specific and actionable question:

> **Can Triple-A performance predict which Tacoma Rainiers player will be the Seattle Mariners' next call-up?**

The project was then redesigned around actual Tacoma-to-Seattle transaction history, with a chronological 2026 holdout used to evaluate the final model.

---

## Project Structure

```text
RainierToMarinerPredictiveModel/
├── data/
│   ├── players.csv
│   ├── training.csv
│   ├── model_data.csv
│   ├── mariners_transactions.csv
│   ├── tacoma_callups.csv
│   ├── callup_model_data.csv
│   ├── 2026_rainiers_predictions.csv
│   └── 2026_callup_predictions.csv
├── visualizations/
│   ├── model_coefficients.png
│   └── 2026_callup_predictions.png
├── analysis.py
├── callups.py
├── callup_model_data.py
├── callup_model.py
├── model.py
├── README.md
├── requirements.txt
└── .gitignore