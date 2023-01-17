# Visualization - plotting

---

---

## Common setup

It is common to setup in the beginning of a notebook:

```python
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

### Background color :

```python
fig = plt.figure(1)
fig.patch.set_facecolor('white')
```

### Title, Label

```python
plt.ylabel('log of fqz')
plt.xlabel('channel val')
plt.title(title)
```

### Take logarithm of a axis

```python
plt.yscale("log")
```

---

## Subplot related

### Subplot in short:

```python
fig = plt.figure(1)
# 211 = 2 row, 1 col, i-th
plt.subplot(211)
plt.subplot(212)
```

### Align x, y axis of subplots:

```python
x = np.min(all_x_data)
y = np.max(all_y_data)
plt.ylim(x, y)
```

### Spacing between subplots:

```python
# set better padding
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
```

---

## Advanced Issues

### Matplotlib not showing images in vscode / notebook :

In vscode jupyter / normal jupyter notebook, sometimes matplotlib cannot draw image.

Try to put one of below in the 1st line of notebook

Ref: [https://github.com/matplotlib/matplotlib/issues/14534](https://github.com/matplotlib/matplotlib/issues/14534)

```python
# %matplotlib inline - Figures are shown as static png images (optionally svg if configured)
# %matplotlib notebook or %matplotlib nbagg - Interactive Figures inside the notebook
# %matplotlib widgets - - Interactive Figures inside the notebook (requires jupyter-matplotlib to be installed)
# %matplotlib tk or %matplotlib qt etc. - GUI windows show the figure externally to the notebook with the given interactive backend
```

---

## Pandas plotting

### Specific case - visualize TF `model.fit` history object:

```python
# The history is from model.fit like this:
# history = conv_model.fit(xxxx)

# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```

---

## Image

### Store the result into image

```python
plt.savefig("/xxxxxx/plt_result.png")
```

### Draw the image with a numpy 2d matrix:

(it represents an image, each value is between 0,255 (Grey scale)) 

```python
# e.g.  train_images.shape : (100, 100)
# Each value is 0~255 (because grey scale)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

### Plot multiple images in a grid:

```python
# import numpy as np
# images_list = np.random.rand(20, 64, 64, 3) * 255

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, img in enumerate(images_list):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(img.astype("int"))
		# Optional
		plt.title(label_list)
    plt.axis("off")
```

---

## General plotting

### Plot Cost during training (1-line):

```python
# Plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()
```

### Other type of plotting:

```python

# Scatter
plt.scatter(x, y, label = "something", c="red")
```

### Plot multiple accuracy chart:

```python
# figure 1
plt.figure(1)

# subplot of 2 col & 1 row & 1st position
plt.subplot(211)
# Plot the train accuracy
plt.plot(np.squeeze(train_acc))
plt.ylabel('Train Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))

# subplot of 2 col & 1 row & 1st position
plt.subplot(212)
# Plot the test accuracy
plt.plot(np.squeeze(test_acc))
plt.ylabel('Test Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show()
```

### Plot multiple line:

```python
# create data
x = [10,20,30,40,50]
y = [30,30,30,30,30]
  
# plot lines
plt.plot(x, y, label = "line 1")
plt.plot(y, x, label = "line 2")
plt.legend()
plt.show()
```

### Plot decision boundary:

> [https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb](https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb)
> 

It is very useful and common during Coursera courses

```python
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
```

---

## 3D plotting

Ref to this post - [https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d](https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d)

```python
import plotly.express as px
import pandas as pd

# df['cluster_label'] is like 0,1,2
"""df looks like:
x0	x1	x2	cluster_label
-0.366353	1.022466	1.166899	2
-1.179214	1.318905	1.047407	2
0.346441	-1.360488	-0.417740	1
"""

# Visualize our data
colors = px.colors.sequential.Plasma
colors[0], colors[1], colors[2] = ['red', 'green', 'blue']
fig = px.scatter_3d(df, x='x0', y='x1', z='x2', color=df['cluster_label'].astype(str), color_discrete_sequence=colors, height=500, width=1000)
fig.update_layout(showlegend=False,
                  scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                    center=dict(x=0, y=0, z=-0.1),
                                    eye=dict(x=1.5, y=-1.4, z=0.5)),
                  margin=dict(l=0, r=0, b=0, t=0),
                  scene=dict(xaxis=dict(backgroundcolor='white',
                                        color='black',
                                        gridcolor='#f0f0f0',
                                        title_font=dict(size=10),
                                        tickfont=dict(size=10)),
                             yaxis=dict(backgroundcolor='white',
                                        color='black',
                                        gridcolor='#f0f0f0',
                                        title_font=dict(size=10),
                                        tickfont=dict(size=10)),
                             zaxis=dict(backgroundcolor='lightgrey',
                                        color='black', 
                                        gridcolor='#f0f0f0',
                                        title_font=dict(size=10),
                                        tickfont=dict(size=10))))
fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
fig.show()
```

---

## Seaborn

### Setup the color tone

```python
import seaborn as sns
PALETTE=['lightcoral', 'lightskyblue', 'gold', 'sandybrown', 'navajowhite',
        'khaki', 'lightslategrey', 'turquoise', 'rosybrown', 'thistle', 'pink']
sns.set_palette(PALETTE)
```

### Scatter plot:

```python
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)
```