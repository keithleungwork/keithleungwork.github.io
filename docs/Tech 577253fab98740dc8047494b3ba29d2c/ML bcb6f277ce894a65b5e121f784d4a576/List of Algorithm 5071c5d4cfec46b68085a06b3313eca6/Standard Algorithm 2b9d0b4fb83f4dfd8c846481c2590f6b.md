# Standard Algorithm

---

## Linear Models :

### Linear Regression(Ordinary Least Square)  :

[https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)

- Basic algorithms in ML.
- To perform classification with generalized linear models, see [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) instead.
- There are a lot of kinds of linear models. But the fundamental is, output `y` is linear to features `x` (i.e. no x^2…etc)

Simplest Linear Regression :

```python
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(
	[[0, 0], [1, 1], [2, 2]],
	[0, 1, 2]
)

# The coefficient
reg.coef_
# output : array([0.5, 0.5]) 
```

### Ridge Regression :

- A Linear regression model with L2 regularization (L2-norm)
- An explanation - [https://medium.com/@minions.k/ridge-regression-l1-regularization-method-31b6bc03cbf#:~:text=Regularization is a technique that,the parameters regular or normal](https://medium.com/@minions.k/ridge-regression-l1-regularization-method-31b6bc03cbf#:~:text=Regularization%20is%20a%20technique%20that,the%20parameters%20regular%20or%20normal).

```python
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

reg.coef_
# array([0.34545455, 0.34545455])
reg.intercept_
# 0.13636...
```

*Q:* Why adding L2-norm ?

*Ans:* 

Simple Linear regression (Ordinary Least Square) is easy to overfit. It performs well in train data but worse in test data and new future data.

We use Ridge Regression to find a new line that doesn’t fit the training data well. In other words, we introduce a small bias into how the new line is fit to data and in return we obtain a significant drop in variance. By starting with a slightly worse fit, Ridge regression can provide better long-term predictions.

*Q:* How to choose the alpha value ?

*Ans:*

The higher the alpha, the less y being sensitive to feature x

To find the optimum values of lambda that results in lowest variance, we use “10-fold Cross Validation Method”. (quoted from [here](https://medium.com/@minions.k/ridge-regression-l1-regularization-method-31b6bc03cbf#:~:text=Regularization%20is%20a%20technique%20that,the%20parameters%20regular%20or%20normal.))

```python
import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
# RidgeCV(alphas=array([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01,
#      1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]))

reg.alpha_
# 0.01

# ???? So 0.01 is the best alpha for the given X training data ???
```

### Ridge Classification :

The **`[RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier)`** can be significantly faster than e.g. **`[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)`** with a high number of classes because it can compute the projection matrix only once.

---

## Decision Tree :

[https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

The most basic single Decision Tree algorithm. From 1 parent root to N children leaf nodes.

```python
# Or DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
# View the tree visually
tree.plot_tree(clf) 
# Or in textual way
r = export_text(clf, feature_names=iris['feature_names'])
print(r)
```

Pros:

- Simple, interpretable, can visualize
- Speed can be fast, depending on the tree depth

Cons:

- Very easy to become overfitting
    - Solution: Pruning, or set min num of data points within each leaf nodes (i.e. to support)
- Unstable, because a small new data can result in a totally different tree
- Output is NOT continuous nor smooth
- Data is easy to be biased if some classes are dominate.
    - Solution: balance the training data across classes

---

## Random Forest :

It is like ensemble of multiple decision trees, each of tree represent 1 component.

```python
from sklearn.ensemble import RandomForestRegressor
```

---