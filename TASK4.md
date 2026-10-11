# Task 4 — Model improvement

This contribution implements Task 4 only. Final model selection (Task 5), test
predictions (Task 6), and the final submission README remain for the team.

## Reproduce

Validated with Python 3.13.3. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_notebook.py
```

Alternatively, open `challenge1.ipynb` in your IDE, select `.venv/bin/python` as
the notebook environment, and run all cells from a fresh kernel. Keep
`task4_modeling.py` alongside the notebook. The runner uses the active Python
interpreter for its kernel and saves executed outputs back into the notebook.
The complete searches and nested checks can take tens of minutes; hardware
affects runtime.

## Experiments

- Random Forest: 24 configurations of depth, minimum leaf size, feature sampling,
  and engineered features on/off; 300 trees and a fixed random seed.
- Gradient Boosting: 24 configurations of learning rate, number of trees, depth,
  and engineered features on/off; minimum leaf size fixed at five.
- Histogram Gradient Boosting: 24 configurations of learning rate, leaf count,
  L2 regularization, and engineered features on/off; 500 iterations without
  internal early stopping.
- Every configuration uses Task 3's shuffled five-fold split and original-dollar
  RMSE. All preprocessing is fitted within each training fold.
- Feature candidates are row-wise size/room ratios, log area, and rotated
  coordinates. No target encoding, external data, or test statistics are used.
- The leading family's complete grid is rerun in a three-fold inner search for
  each of the original five outer folds. This evaluates its tuning procedure
  without using the outer-fold prices during the inner search. Family selection
  has already seen the training CV results, so this conditional check retains
  family-selection bias. It is not a fully nested selection across all families.
- Because the initial boosting winner was at the depth and tree-count boundaries,
  a follow-up 16-candidate grid tests depth 4/6, 800/1,200 trees, learning rate
  0.05/0.1, and features off/on. Its own conditional nested check is reported
  separately. The follow-up ranges were informed by earlier CV results, so they
  also retain adaptive-selection bias.

The finite grids are systematic experiments, not a claim of finding a global
optimum. Best Grid Search scores are selection scores, not unbiased estimates of
hidden-test performance. Fold standard deviations are descriptive, not confidence
intervals. See the notebook for the completed tables and interpretation.

## Handoff

`task4_results/` contains complete candidate tables, baseline improvements,
feature ablations, nested fold results, and `summary.json` with selected
parameters. These artifacts are outputs, never cached inputs to the notebook.

For Task 5, `searches[name].best_estimator_` and `searches[name].best_params_`
are available after executing Task 4. The team can reconstruct a configuration
using `task4_modeling.make_pipeline` and the parameters in `summary.json`.
Grid Search refits each winning candidate on the full training data, but Task 4
does not declare a final competition model or generate test predictions.

Preserve fixed estimator settings as well as searched settings. For an initial
grid candidate, reconstruct with `pipeline, _ = search_spaces()[name]` and then
`pipeline.set_params(**params)`, where `params` is that family's entry in
`summary.json`. This retains, for example, boosting's fixed minimum leaf size of
five and the forest's 300 trees. The refinement also stores the complete model
constructor parameters in `refinement_summary.json`.

The follow-up additionally exposes `refined_search`, `refinement_summary`, and
`task4_candidates`. Check both the original and expanded results; more complexity
does not guarantee a lower validation error.

## Observed results

| Family | Baseline CV RMSE | Best tuning CV RMSE |
| --- | ---: | ---: |
| Random Forest | 423.63 | 412.61 |
| Gradient Boosting | 460.76 | 392.42 |
| Histogram Gradient Boosting | Not evaluated as a baseline | 399.27 |

The initial Gradient Boosting grid's conditional nested mean outer RMSE is
391.45 (fold standard deviation 29.82). The expanded grid keeps the same winning
configuration and gives a conditional nested mean of 392.70 (standard deviation
29.22). The winner uses engineered features, 800 trees, learning rate 0.1, depth
4, and minimum leaf size 5. Boosting improves its baseline tuning score by 14.83%,
but increases the train/CV gap; this is evidence for Task 5, not a final selection.

Verification: the original notebook plus initial Task 4 searches and nested check
ran from a fresh Jupyter kernel. The follow-up training and nested check ran in a
fresh Python process using the identical function called by the notebook; its
actual console outputs are saved in the notebook. The comparison cell was also
executed, and every code cell passed syntax and notebook-format validation.
Studio handling, input immutability, unseen-category prediction, estimator cloning,
and initial grid sizes were checked. No final test predictions were generated.
