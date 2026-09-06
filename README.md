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

### Real-World Validation

The models surfaced several players who subsequently reached Seattle.

- **Colt Emerson** was ranked **#2 at 85.1%** by the original MLB-arrival model and was promoted to Seattle on May 17, 2026.
- **Lazaro Montes** was ranked **#1 at 89.5%** by the original MLB-arrival model and was promoted on September 1, 2026.
- In the final Tacoma-to-Seattle call-up model, **Michael Arroyo ranked #3 among eligible Tacoma hitters** immediately before his September 6, 2026 promotion.

These results do not prove that the model can predict front-office decisions, but they provide useful real-world evidence that the performance signals identified by the models can align with actual player movement.

### What I Found

The Random Forest achieved a **0.701 ROC-AUC** on a chronological 2026 holdout, outperforming Logistic Regression at 0.660.

The model was therefore useful as a **ranking tool**, even though its scores should not be interpreted as calibrated probabilities.

![2026 Tacoma Call-Up Predictions](visualizations/2026_callup_predictions.png)

---

## Project Overview

This project started with a broader question:

> Can a player's Triple-A performance be used to predict whether they will reach MLB in the following season?

I began by building a model using historical Triple-A performance from across Minor League Baseball and determining whether each player appeared in MLB the following season.

The project then evolved into a more specific and actionable question:

> Can Triple-A performance help predict which Tacoma Rainiers player will be the Mariners' next call-up?

This second iteration uses actual Tacoma-to-Seattle transaction history and focuses specifically on call-up prediction.

---

## Project Structure

```text
RainierToMarinerPredictiveModel/
├── data/
├── visualizations/
├── analysis.py
├── model.py
├── callups.py
├── callup_model_data.py
├── callup_model.py
├── README.md
├── requirements.txt
└── .gitignore