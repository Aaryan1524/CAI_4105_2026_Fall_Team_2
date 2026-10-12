# Challenge 1 — Apartment Rental Price Prediction

For this challenge, our team built a model to predict monthly apartment rent
using apartment characteristics and location. We used the 7,398 provided training
rows to develop and compare our models. We kept `price` as the target and removed
`ID` from the predictors.

We used median imputation for numerical features and most-frequent imputation
with one-hot encoding for categorical features. We put preprocessing inside each
model's pipeline so it was fitted only on the training portion of each fold.
We used only the provided datasets and saved the test data for final predictions.

## Validation and models

We compared Linear Regression, Decision Tree, Random Forest and Gradient Boosting
using the same shuffled five-fold cross-validation with `random_state=42`.
We chose RMSE because it measures prediction error in the same units as monthly
rent and gives more weight to larger errors. We also compared training and
validation errors to look for overfitting or limited model flexibility.

We used Grid Search to tune Random Forest and Gradient Boosting, testing eight
parameter combinations for each. Our results are shown below. Since we used the
CV scores to choose parameters and select our model, the best score may be
optimistic. We do not know our hidden-test RMSE yet.

| Model | Baseline CV RMSE | Tuned CV RMSE |
| --- | ---: | ---: |
| Linear Regression | 526.76 | — |
| Decision Tree | 562.39 | — |
| Random Forest | 423.63 | 415.00 |
| Gradient Boosting | 460.45 | **398.68** |

We selected **tuned Gradient Boosting** because it had the lowest mean CV RMSE,
**398.68**, improving its baseline by **13.42%**. Our selected settings were 800
trees, learning rate 0.1, maximum depth 4, minimum leaf size 5 and seed 42.

This model had a larger training/validation gap than tuned Random Forest and is
harder to interpret than Linear Regression or a single tree. We still chose it
because its validation error was lower and its training time was manageable for
our dataset. We discuss these tradeoffs in the notebook. We then fitted the
selected pipeline on all training rows to generate our final predictions.

## Run the notebook

Our final notebook was run with Python 3.13.3. The package versions are listed in
`requirements.txt`. To set up the environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open `challenge1.ipynb` in Antigravity or another notebook-capable IDE, select
`.venv/bin/python` as the Python environment, and run every cell from a fresh
kernel. Keep the supplied training and test CSV files alongside the notebook.
The full workflow takes several minutes, depending on hardware.

The final cell creates `predictions.csv` with the columns `ID` and `price` for all
2,467 test rows in their original order. We checked that the file preserves every
test ID, has no missing predictions and contains finite positive prices.
All modeling code is included in the notebook, so no additional Python files
are needed.

## Submission

Our submission includes `challenge1.ipynb`, `README.md`, `predictions.csv` and
`requirements.txt`. The provided training and test datasets are also in our
repository so the workflow can be reproduced. Each team member is responsible
for submitting their own peer-review form in Canvas.
