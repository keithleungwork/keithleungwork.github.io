# Feature Engineering

> It is also called feature extraction
> 

Purpose: Turn preprocessed text into → numeric vector (i.e. text representation?) that can be fed into ML 

## Feature engineering in ML VS in DL

### In ML:

Pros:

We handcraft the function to do this.

i.e. it remains interpretable, we can explain the feature correlation

Cons:

Handcrafted function is a bottleneck sometimes. It affects the performance

Also, the wrong choice of feature could lead to big harm on the model.

### In DL:

Pros:

Pre-processed raw data is directly fed into DL model.

So the model performance is higher

DL model “learn” the features from data.

Cons:

HOWEVER, it learns all features from data, it is hard to tell the correlation then.

i.e. it loses interpretability.

---