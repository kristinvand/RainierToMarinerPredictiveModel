# Rainier to Mariner Predictive Model

## Can Triple-A performance help predict which Tacoma Rainiers player will be the Mariners' next call-up?

This project explores whether performance in Triple-A can provide a useful signal for identifying players who are likely to reach MLB.

The project started with a broader question:

> Can a player's Triple-A performance predict whether they will reach MLB in the following season?

That analysis established a baseline relationship between Triple-A performance and MLB promotion. The project was then narrowed to a more actionable baseball-operations question:

> Can Triple-A performance help identify which Tacoma Rainiers player will be the Mariners' next call-up?

The final model uses historical Tacoma Rainiers performance and Seattle Mariners transaction data to rank current Tacoma players by their likelihood of receiving an MLB call-up.

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