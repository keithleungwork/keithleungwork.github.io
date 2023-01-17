# PCA (Principal Component Analysis)

Comment: Developed in 1933……, seriously ?

---

---

## Implementation:

### To implement with scikit:

```python
from sklearn.decomposition import PCA

# only return 2 components
pca = PCA(n_components=2)
pca.fit(flat_x) # the input features shape should be: (samples, features)

print(pca.explained_variance_ratio_)
# out: [0.58640506 0.06278766]

# transform data for visualization later
X_r = pca.transform(flat_x)
X_r.shape
# out: [samples, 2]

# OR, you can fit and transform at the same time
X_r = pca.fit_transform(flat_x)
```

---

## What is PCA?

*TLDR*: 

A ***linear*** technique, to reduce the ***number of variables*** in a dataset, while preserving as much information as possible. (i.e. reduce the dimensionality of large data sets)

- NORMALLY, reducing num of var → accuracy drops
BUT with PCA, it trades a little accuracy for simplicity.
- Benefit:
    - easier to explore & visualize & analyze dataset
    - faster process for ML algorithm
    

Some visual illustration:

- The original 3-dimensional data set. The red, blue, green arrows are the direction of the ***first, second, and third principal components***, respectively. Image by the author.
    
    ![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled.png)
    
    [Image from [here](https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d)]
    
- After PCA, 3-dim is reduced into 2-dim:
    
    ![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled%201.png)
    

---

## How does PCA work?

1. it understand the datasets first, by math, e.g. variance….etc
    - Goal: understand how important it is for each variables (i.e. does it hold more info?)
    - e.g. The greater the variance, the more the information. Vice versa.
    - Reason: e.g. every data hold a variable of similar value(i.e. small variance), we cannot tell the difference of them by this variable.
    
2. Summarizing data (i.e. reduce dim)

It is done by finding those ***Principal components.***

Let’s use an example below to explain:

When we look closer at our data, the maximum amount of variance lies not in the x-axis, not in the y-axis, but a diagonal line across. The second-largest variance would be a line 90 degrees that cuts through the first.

![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled%202.png)

Then use the Principal Components as axis (PC1, PC2…etc)

![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled%203.png)

In this example, PC1 alone can capture the total variance of Height and Weight combined. Since PC1 has all the information, we can be very comfortable in removing PC2 and know that our new data is still representative of the original data.
(Here is an ideal case)

What about PC2, …etc is not 0% of info?

In a real-world situation, PC2 is not always 0% of info like in the above example.

Performing a PCA will give us N number of principal components, where N is equal to the dimensionality of our original data. Then generally choose the ***least number*** of principal components that would explain the most amount of our original data.

E.g. by looking at the variance of each Principal Component. 
Let’s say we want to keep 90% of info (i.e. 0.9 variance in total), we only need PC1 & PC2 here.

![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled%204.png)

What data is lost during PCA?

Here is the illustration, the red line distance is the data we lost if we only keep this PC1

![Untitled](PCA%20(Principal%20Component%20Analysis)%206f6d005a20a14022a296f1433a97d9a0/Untitled%205.png)

When reducing dimensions with PCA, it changes the distances of our data. (the distance between each others)

---

## Resources

- WIki -  [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis):
- Scikit-learn - [https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
- Simple explanation - [https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d](https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d)
    - a colab to illustrate it - [https://colab.research.google.com/drive/1RC_XulRdrqpYRq4h8pRl22cfg9a-_FvS?usp=sharing](https://colab.research.google.com/drive/1RC_XulRdrqpYRq4h8pRl22cfg9a-_FvS?usp=sharing)
- ****Image Classification with Principal Component Analysis -**** [https://www.christopherlovell.co.uk/blog/2016/07/04/image-pca-deckchair.html](https://www.christopherlovell.co.uk/blog/2016/07/04/image-pca-deckchair.html)
    - Q:  Why a monochrome image has color like that?
- Use PCA in Image compression - [https://towardsdatascience.com/image-compression-using-principal-component-analysis-pca-253f26740a9f](https://towardsdatascience.com/image-compression-using-principal-component-analysis-pca-253f26740a9f)
- Use PCA for MNIST dataset demonstration - [https://builtin.com/data-science/tsne-python](https://builtin.com/data-science/tsne-python)