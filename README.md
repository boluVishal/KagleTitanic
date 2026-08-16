# Kaggle — Titanic: Machine Learning from Disaster

My solution for the classic Kaggle Titanic competition. The task is binary classification: predict which passengers survived based on features like age, sex, ticket class, and fare.

## Files

| File | What it is |
|---|---|
| `titanic.py` | Feature engineering + model training |
| `train_2.csv` | Training data |
| `test_2.csv` | Test data |
| `gender_submission.csv` | Baseline submission (predict all females survive) |

## Running it

```bash
python titanic.py
```

Requires `pandas`, `numpy`, and `scikit-learn`.

## Competition

The Titanic dataset is a standard entry point for classification in ML. Details at [kaggle.com/c/titanic](https://www.kaggle.com/c/titanic).
