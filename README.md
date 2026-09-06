# Rainier to Mariner

## Who's Next?

Can Triple-A performance help predict which Tacoma Rainiers player is next to receive a Seattle Mariners call-up?

This project uses MLB Stats API data, historical Mariners transactions, and machine learning to explore that question.

The project started with a broader question:

> **Can a player's Triple-A performance predict whether they will reach MLB the following season?**

After building that model, I narrowed the project to the more specific baseball operations question:

> **Who's next?**

---

## Final Results

The final call-up model was trained on **2021–2025** and tested on the completely held-out **2026 season**.

| Model | 2026 ROC-AUC |
|---|---:|
| **Random Forest** | **0.701** |
| Logistic Regression | 0.660 |

The Random Forest performed best and was used for the final rankings.

The model produces a **ranking score**, not a calibrated probability.

### 2026 Top 10

| Rank | Player | Model Score |
|---:|---|---:|
| 1 | Alejo Lopez | 0.488 |
| 2 | Nick Raposo | 0.406 |
| 3 | **Michael Arroyo** | **0.319** |
| 4 | Ryan Bliss | 0.252 |
| 5 | Blake Rambusch | 0.243 |
| 6 | Victor Labrada | 0.237 |
| 7 | Andy Ibáñez | 0.218 |
| 8 | Victor Robles | 0.204 |
| 9 | Connor Joe | 0.192 |
| 10 | Spencer Packard | 0.169 |

![2026 Tacoma Call-Up Predictions](visualizations/2026_callup_predictions.png)

---

## Real-World Validation

The project produced some interesting results when compared with actual player movement.

Across the different versions of the model:

- **Colt Emerson** ranked #2 in the original MLB-arrival model with an 85.1% model output and was later promoted to Seattle.
- **Lazaro Montes** ranked #1 in the original MLB-arrival model with an 89.5% model output and was later promoted to Seattle.
- **Michael Arroyo** ranked #3 among eligible Tacoma hitters in the final call-up model immediately before his promotion to Seattle.

These results don't prove that the model can predict front-office decisions, but they provide an interesting real-world check on whether the model is identifying useful signals.

---

# The Original Question

The project originally asked:

> **Can a player's Triple-A performance predict whether they will reach MLB the following season?**

I collected Triple-A hitting data for:

- 2021 → 2022 MLB
- 2022 → 2023 MLB
- 2023 → 2024 MLB
- 2024 → 2025 MLB

A player was labeled:

```text
mlb_next_season = 1
```

if they recorded MLB hitting statistics the following season.

This produced **4,336 player-seasons**:

- 1,642 reached MLB
- 2,694 did not

---

# Baseline Model

The baseline model used:

- Age
- Plate appearances
- Batting average
- OBP
- SLG
- Home runs
- Walk rate
- Strikeout rate
- Stolen bases
- BABIP

Players with fewer than 100 plate appearances were excluded.

I compared Logistic Regression and Random Forest.

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 0.737 |
| Precision | 0.716 |
| Recall | 0.702 |
| ROC-AUC | **0.797** |

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 0.707 |
| Precision | 0.682 |
| Recall | 0.672 |
| ROC-AUC | 0.773 |

Logistic Regression performed best on the original problem.

---

# Player Development Experiment

I then tested whether **year-over-year improvement** added useful information.

Additional features included changes in:

- OPS
- OBP
- SLG
- Walk rate
- Strikeout rate

The results were mixed.

| Season | Baseline ROC-AUC | Development ROC-AUC |
|---|---:|---:|
| 2023 | 0.788 | **0.796** |
| 2024 | **0.772** | 0.764 |

Because the development features did not consistently improve performance, I kept the simpler feature set.

---

# Changing the Question

The original model answered:

> **Who is likely to reach MLB next season?**

But that's different from:

> **Who is the Mariners' next call-up?**

Call-ups depend on more than player performance. Roster openings, injuries, positional needs, and organizational decisions all matter.

So I changed the target to:

```text
called_up = 1
```

if a Tacoma player was promoted to Seattle during the following week.

---

# Building the Call-Up Dataset

I collected Mariners transaction data from 2021–2026 and identified Tacoma-to-Seattle transactions.

I then created weekly Tacoma player snapshots using cumulative performance up to each point in the season.

The final dataset contained:

- **4,281 observations**
- **110 call-ups**
- About **2.6% positive observations**

Because call-ups are relatively rare, accuracy isn't a useful primary metric.

Instead, I focused on:

- ROC-AUC
- Precision
- Recall
- Precision@K

---

# Final Model

For the final evaluation, I used a chronological split:

```text
Training: 2021–2025
Testing: 2026
```

This prevents the model from seeing future-season outcomes during training.

### Results

| Model | 2026 ROC-AUC |
|---|---:|
| Logistic Regression | 0.660 |
| **Random Forest** | **0.701** |

The Random Forest was selected for the final rankings.

---

# What I Learned

### 1. Performance contains useful signal

Triple-A offensive statistics can provide meaningful information about future MLB arrival and call-up likelihood.

### 2. More features aren't always better

Year-over-year development features did not consistently improve the model.

### 3. Validation matters

A chronological holdout provides a much more realistic test than randomly mixing past and future seasons.

### 4. Ranking may be more useful than classification

For a baseball operations use case, the goal may be less about predicting a yes/no outcome and more about identifying players who deserve additional attention.

---

# Limitations

This is a **proof-of-concept**, not a production player evaluation system.

The model does not capture things like:

- Roster availability
- Injuries
- Position needs
- Defensive ability
- Scouting information
- Coaching evaluations
- Organizational plans

The call-up dataset is also relatively small and highly imbalanced.

The 2026 holdout provides a useful future-season test, but additional seasons would be needed to determine whether the results generalize.

---

# Future Improvements

Possible next steps include:

- Add player position and defensive value
- Add prospect rankings
- Incorporate Mariners roster needs
- Add Statcast data
- Add rolling performance trends
- Test gradient boosting models
- Improve probability calibration
- Use additional future seasons for validation
- Explore time-to-call-up models

---

# Project Structure

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
```

---

# Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- MLB Stats API
- Git
- GitHub

---

# Running the Project

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the main scripts:

```bash
python analysis.py
python model.py
python callups.py
python callup_model_data.py
python callup_model.py
```

---

## Project Goal

The goal of this project was not to build a perfect prediction system.

It was to explore whether publicly available baseball data could be turned into a useful tool for identifying players who may be approaching their next level.

The project ultimately asks a simple question:

> **Who's next?**
