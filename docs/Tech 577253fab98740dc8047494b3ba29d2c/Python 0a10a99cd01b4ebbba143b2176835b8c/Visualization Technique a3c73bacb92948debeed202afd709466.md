# Visualization Technique

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

---

## Issues

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

## Image

Display a numpy 2d matrix

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

---

## General plotting

### Plot decision boundary:

> [https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb](https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb)
> 

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

Setup the color tone

```python
import seaborn as sns
PALETTE=['lightcoral', 'lightskyblue', 'gold', 'sandybrown', 'navajowhite',
        'khaki', 'lightslategrey', 'turquoise', 'rosybrown', 'thistle', 'pink']
sns.set_palette(PALETTE)
```