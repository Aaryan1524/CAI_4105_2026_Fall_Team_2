# Challenge 1 — Apartment Rental Price Prediction

The notebook completes Tasks 1–6 using the supplied 7,398-row training dataset.
`price` is the target and `ID` is excluded from predictors. Numerical features are
median-imputed; categorical features are imputed and one-hot encoded inside each
model pipeline. The test dataset is used only for final predictions.

## Validation and models

All models and Grid Searches use the same shuffled five-fold cross-validation
with `random_state=42`. RMSE in monthly-price units is the primary metric. The
notebook reports training RMSE, validation RMSE, fold spread, gaps and fit time.
Grid Search tunes Random Forest and Gradient Boosting with eight configurations
each. The best search scores are tuning/selection estimates and may be optimistic;
hidden-test performance is not yet known.

| Model | Baseline CV RMSE | Tuned CV RMSE |
| --- | ---: | ---: |
| Linear Regression | 526.76 | — |
| Decision Tree | 562.39 | — |
| Random Forest | 423.63 | 415.00 |
| Gradient Boosting | 460.45 | **398.68** |

The final model is **tuned Gradient Boosting**, with 800 trees, learning rate 0.1,
depth 4, minimum leaf size 5 and seed 42. It has the lowest observed validation
RMSE and improves its baseline by 13.42%. Its larger training/validation gap and
lower interpretability are discussed in the notebook; training cost is manageable
for this dataset. The selected pipeline is explicitly refitted on all training
rows before generating predictions. No external datasets are used.

## Run the notebook

Validated with Python 3.13.3; dependencies are pinned in `requirements.txt`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open `challenge1.ipynb` in Antigravity or another notebook-capable IDE, select
`.venv/bin/python` as the Python environment, and run every cell from a fresh
kernel. Keep the supplied training and test CSV files alongside the notebook.
The full workflow takes several minutes, depending on hardware.

The final cell writes `predictions.csv` with exactly `ID` and `price`, containing
all 2,467 test rows in their original order. It checks IDs, completeness, finite
positive prices and the saved file's format. No additional project code is needed.

## Submission

The required deliverables are `challenge1.ipynb`, `README.md`, `predictions.csv`
and `requirements.txt`, with the supplied data available for reproduction.
Maintain the final files in the team's GitHub repository and upload the requested
files to Canvas. Each team member must submit their own peer-review form separately.
