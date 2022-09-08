# Ensemble Model

From scikit learn:

> The goal of **ensemble methods** is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.
> 

---

---

## Concepts

- Meta-model - a model we train to average the result of other models

## Ref

- [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)

## Ensemble methods

- Averaging
    - Build models independently in training
    - In prediction, average their outputs
    - Pros :
        - **Usually** better than any single model inside this ensemble model, because the variance is reduced.
    - Example :
        - [Bagging methods](Ensemble%20Model%20b8a30bf59aec436d8f98d75b85fc31dc.md)
        - [Forests of randomized trees](Ensemble%20Model%20b8a30bf59aec436d8f98d75b85fc31dc.md)
- Boosting
    - Build base models sequentially ???
    - Pros :
        - Combine weak models to become powerful model
- Stacking
    - advantage of stacking is that it can benefit even from models that don't perform very well
- Blending
    - require that models have a similar, good predictive power.
    - Blending ensembles are a type of stacking where the meta-model is fit using predictions on a `holdout validation dataset` instead of `out-of-fold predictions`.
    

---

## Bagging Methods

- Bagging methods - [https://scikit-learn.org/stable/modules/ensemble.html#bagging](https://scikit-learn.org/stable/modules/ensemble.html#bagging)

How :

It build several models in a black-box
Each of the models uses a subset of training data
At last, their predictions are aggregated to form a final decision

It can be easily implemented with scikit like this:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
# Implement bagging with K-neighbors algorithm for example
bagging = BaggingClassifier(
	KNeighborsClassifier(),
	# Max drawn from 50% of samples
	max_samples=0.5,
	# Max drawn from 50% of features (i.e. fields, columns...)
	max_features=0.5)
```

Pros :

Reduce variance of the base models

Work best with strong & complex models

Example :

- [Single estimator versus bagging: bias-variance decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py)

---

## Random Forest Classifier

- scikit-learn forest class - [https://scikit-learn.org/stable/modules/ensemble.html#forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)

How :

Similar to bagging method, the only difference is RandomForest is specialized for Decision Trees.

A diverse set of classifiers is created by introducing randomness in the classifier construction.

The final decision is averaged among predictions made by sub classifiers.

Each decision trees is built from the samples drawn ***with replacement.***

Pros:

Decrease the variance of the forest estimator