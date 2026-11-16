# Challenge 2 — Online Shopper Purchasing Intention

For this challenge, our team built a classifier that predicts whether an online
shopping session ends in a purchase (`Revenue`). We used only the 9,247 training
sessions to develop and validate our models and used the 3,083 test sessions only
to generate final predictions. `ID` is excluded from the predictors. Only 15.5% of
sessions convert, so we evaluated models with F1 for `Revenue=True` instead of
accuracy.

## Approach

**Preparation.** Numeric features are standardized. `Month`, `VisitorType`,
`OperatingSystems`, `Browser`, `Region` and `TrafficType` are one-hot encoded. The
last four are stored as integers, but they are category codes, not quantities.
`Weekend` passes through unchanged. All preprocessing sits inside each model's
pipeline, so it is fitted on training folds only.

**Validation.** We split the training data into a development set (85%) and a
holdout set (15%), stratified by `Revenue`. All model comparison and tuning used the
same 5-fold stratified cross-validation on the development set (`random_state=42`).
F1 for the positive class is the primary metric. We also report precision, recall,
ROC AUC, confusion matrices and the train/validation F1 gap to check for overfitting.
The holdout was used once, for the final model only.

## Models evaluated

Starting configurations, mean 5-fold CV F1 on the development set:

| Model | CV F1 |
| --- | ---: |
| Gradient Boosting | 0.673 |
| Random Forest | 0.668 |
| Support Vector Machine | 0.601 |
| Decision Tree | 0.596 |
| Logistic Regression | 0.522 |
| K-Nearest Neighbors | 0.480 |
| Naive Bayes | 0.305 |

We used Grid Search to tune the two strongest models. Each classifier's probability
threshold was then tuned with nested cross-validation: inside every training fold,
an inner 5-fold split chooses the F1-maximizing threshold, so the reported scores
never use the folds that chose it.

| Model | Decision rule | CV F1 | Train/val gap |
| --- | --- | ---: | ---: |
| Gradient Boosting | tuned threshold (~0.30) | **0.694** | 0.053 |
| Random Forest | tuned threshold (~0.59) | 0.691 | 0.180 |
| Gradient Boosting | 0.5 | 0.680 | 0.062 |
| Random Forest | 0.5 | 0.679 | 0.144 |

## Final model

**Tuned Gradient Boosting with a nested-CV probability threshold** (200 trees,
learning rate 0.05, max depth 3). It ties Random Forest on CV F1 (the difference is
under the 0.005 margin we fixed before comparing), and it overfits far less: a
train/validation gap of 0.05 against 0.18. It is also quick to fit. `PageValues`
accounts for most of its feature importance, which matches the data exploration.

**Final validation performance:** mean CV F1 **0.694** (fold SD 0.010; precision
0.63, recall 0.77). On the one-time holdout, F1 was **0.636** (bootstrap 95%
interval 0.586–0.684). That is lower than CV suggested, so we expect test F1 to be
closer to the holdout figure. We report the gap rather than tuning against the
holdout.

For the submission, the model and its threshold search are refitted on all 9,247
labeled sessions. The final threshold is 0.338.

## Predictions

`challenge2/predictions.csv` contains `ID` and the predicted `Revenue` (`True` /
`False`) for all 3,083 test sessions in their original order. The notebook checks
that every test ID is present exactly once and that there are no missing values.
About 17% of test sessions are predicted to purchase, slightly above the base rate,
as expected from an F1-optimized threshold.

## Run the notebook

The notebook was run with Python 3.13 using the package versions in the repository's
`requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open `challenge2/challenge2.ipynb`, select the `.venv` interpreter, and run all
cells from a fresh kernel with the working directory set to `challenge2/` (the data
files sit next to the notebook). A full run takes about 7 minutes on a laptop. It
writes the comparison tables and figures (`task3_*`, `task4_*`, `task5_*`) and
`predictions.csv`. All random steps use fixed seeds, so reruns reproduce the same
predictions.
