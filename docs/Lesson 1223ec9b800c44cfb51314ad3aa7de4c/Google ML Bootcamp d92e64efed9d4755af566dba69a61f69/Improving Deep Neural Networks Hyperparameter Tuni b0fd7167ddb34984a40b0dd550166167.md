# Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization

---

# Cite

As in copyright:

> [DeepLearning.AI](https://www.deeplearning.ai/) makes these slides available for educational purposes. You may not use or distribute these slides for commercial purposes. You may make copies of these slides and use or distribute them for educational purposes as long as you cite [DeepLearning.AI](https://www.deeplearning.ai/)  as the source of the slides.
> 
- Slides here are from [DeepLearning.AI](https://www.deeplearning.ai/)

---

# Important Deadline

| Week | Deadline |
| --- | --- |
| 1 | 11 Sep |
| 2 | 18 Sep |
| 3 | 25 Sep |

## Goal

- Practical aspects of deep learning
- Learn about hyperparameter tuning, regularization
- How to diagnose bias, and variants, and advance optimization algorithms, like momentum, armrest, prop, and the ad authorization algorithm

---

# Week 1 - Practical Aspect of Deep Learning

## Train / Dev / Test dataset splitting:

- Training set is used for training
- Dev set (i.e. Hold-out cross validation set), is used to cross validate performance of several models
- Test set is an unbiased dataset to estimate how well your model is
    - Difference with Dev set → Your model actually has bias to the Dev set as well while tuning / cross validation

Dataset Distribution:

- 1st rule - keep all split dataset from the same distribution
    - e.g.
    

Splitting ratio:

- It is common to split it like 70/30 or 60/20/20, IF THE DATASET IS NOT EXTREMELY HUGE
- If the dataset is like > 1,000,000, you can also set a max of dev/test dataset to be e.g. 1000 (i.e.  a constant)

## Bias & Variance:

- High Bias = underfitting
- High Variance = overfitting
- Note!!!! It is possible to be high bias & high variance at the same time.

![Screen Shot 2022-09-07 at 17.56.43.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-07_at_17.56.43.png)

How we determine them from the metrics ?

Assume the bass error is ~ 0% (e.g. Human can almost identify if it is a cat image by 100% accuracy)

| Bias/Variance | Train set Error | Dev set Error |
| --- | --- | --- |
| High bias | 15% | 16% |
| High variance | 1% | 11% |
| Both High!!! | 15% | 30% |

## Solution to high bias / variance :

High Bias 

- It is `**underfitting**` to the training set (i.e. Model cannot “learn” enough from the training set features)
- Use a larger neural network with more layers or units
- Change algorithm

High variance

- It is `**overfitting**` to the training set
- Use more data to train the network
- Regularization
- Change algorithm

## Overfitting/underfitting - Regularization Techniques:

### L2 norm

- Why not L1 norm ?
    - It is said that L1 norm would make parameters matrix to be `Sparse`
    (i.e. many zeros in matrix)
- In calculating Cost func, add the L2 “penality”
- Then in the back propagation, this is added to calculating gradient
- The `lambda` value here is a hyperparameter

![Screen Shot 2022-09-08 at 14.27.46.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.27.46.png)

![Screen Shot 2022-09-08 at 14.30.12.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.30.12.png)

- How does it prevent overfitting ????
    - If lambda is large, the result W will be small (more close to zero)
    - Then, the Z will be fairly small and the activation function is roughly linear
    - Recall from previous course:
        - if Activation function is linear, the whole network tends to behave as a small network.
        - i.e. It acts like a linear network and avoid overfitting
    

![Screen Shot 2022-09-08 at 14.34.10.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.34.10.png)

### Dropout Regularization

- In every propagation, each units are decided randomly (by a configured probability, e.g. 0.5)
- i.e. each propagation uses a different set of units
- Note : the probability can be different between each layers
e.g. last layer would not be overfitting so its probability is set to 1.0
- So, the `probability` here is a hyperparameter

![Screen Shot 2022-09-08 at 14.37.35.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.37.35.png)

- Cautious !!!
The activation value after dropout, need to be divide by the probability
Because it is to keep the expected value to be the same level
- Cautious 2 !!!
When doing a test prediction, do not apply dropout
i.e. ONLY use dropout when “training” the model

![Screen Shot 2022-09-08 at 14.41.03.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.41.03.png)

- How does it work???
    - Somehow it forces the unit to NOT rely heavily on a specific features
    i.e. avoid overfitting

![Screen Shot 2022-09-08 at 14.43.45.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_14.43.45.png)

### Other Regularization Methods

- Data augmentation
    - To avoid heavy data annotation cost / impossible to get more data
    - It increases the data size in a cheaper way
- Early Stopping
    - The loss of dev dataset drops  → increase after certain iteration
    - It usually means overfitting takes place

> An additional important concept here : 
It is better to tune hyperparameters of the same purpose at a certain time, not all together
e.g. Tune those for optimizing Cost Function, then tune those for avoid overfitting…etc
> 

## Optimization - Normalize Input Data:

- Why need to normalize?
    - To speed up the training, because the gradient descent can reach global minima faster

![Screen Shot 2022-09-08 at 17.04.17.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_17.04.17.png)

How does it work ?

One of the normalization method is called - **[standard scalar](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)**

> Standardize features by removing the mean and scaling to unit variance.
> 

### Standard Scalar normalization

- 1st step - Subtract / Zero out the mean of `TRAINING DATA`
    - So the input has a Zero mean

![Screen Shot 2022-09-08 at 16.59.45.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_16.59.45.png)

- 2nd step - normalize the variance
    - e.g. x1 has a larger variance than x2
    

![Screen Shot 2022-09-08 at 17.02.21.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_17.02.21.png)

### Some other common normalization

- Such as divide an image input by 255 if it is all RGB channel

Cautious !!!
Use the calculated mean and sigma in test set or prediction as well
Do not calculate a new one

## Optimization - Vanishing / Exploding Gradient

### What is this problem about  ???

It can be proved by a simplified neural network, and we can know the output could be very large or small, depending on the number of layers.

Note that activation function was assumed to be nothing here. So in fact the output will not be REALLY large, but it is the concept only.

The output is still large enough so the training become very slow.

For example of a sigmoid activation function
The slope near z=`0` or near `1` is extremely small, so the weights near those extremes will **converge much more slowly** to the solution, and having most of them near the center will speed the convergence.

![Screen Shot 2022-09-08 at 17.22.54.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_17.22.54.png)

### Solution - weight initialization

It is just a partial solution but it is still helpful.
By using a correct weight initialization technique.

- For each layer, init the `w` by multiply it with square root of `1/n` , where n is the unit number of ***previous*** layer(or input)
    - Reason - so the initial `w` value is NOT too small or too large
- For **ReLU**, it is said that using `2/n` is a better value, it is called `He initialization`, because it is named for the first author of He et al., 2015
- For **Tanh**, `1/n`, it is also called `**Xavier initialization**`
- For some other way, for example, `2/(n of l-1 + n of l)`

Also, use `np.random.randn`, because it produces numbers in a normal distribution.

> When used for weight initialization, randn() helps most the weights to Avoid being close to the extremes, allocating most of them in the center of the range.
> 

![Screen Shot 2022-09-08 at 17.49.12.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_17.49.12.png)

A conclusion of weight initialization :

| Model | Train accuracy | Problem/Comment |
| --- | --- | --- |
| 3-layer NN with zeros initialization | 50% | fails to break symmetry |
| 3-layer NN with large random initialization | 83% | too large weights |
| 3-layer NN with He initialization | 99% | recommended method |

## Gradient Checking:

It is a debugging step to ensure the gradient descent step you built is correct (especially back propagation)

How it works ???

- Calculate the `gradient` via back propagation
    - hints: convert them into 1-dim array (dW1, dB1, dW2, dB2…)
- Calculate the `approximate gradient` manually, by using theta + - epsilon
For EACH parameters
    - hints: for each comparison(e.g. `dW1` vs `approx_dW1`), keep all other parameters unchanged
    - 1st - get TWO Cost `J` via forward propagation of (***W1+/-epsilon***, B1, W2, B2…)
    - 2nd - calculate the slope(gradient) by $\frac {J(+) - J(-)} { 2 epsilon}$
- At last, compare both 1-dim array `grad` vs `approx_grad`
$\frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 }$
    
    ```python
    numerator = np.linalg.norm( (grad - gradapprox) )
    denominator = np.linalg.norm( grad ) + np.linalg.norm( gradapprox )
    difference = numerator / denominator
    ```
    

![Screen Shot 2022-09-08 at 18.20.15.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-08_at_18.20.15.png)

Note

- It cannot be used in dropout regularization
    - solution - turn off dropout and do a gradient check first
- Only use it for debugging, not in the real training process
- A very rare case
    - maybe grad check is correct when parameters are close to zero
    But incorrect when they are not.
    - So in this case, can do a check after random init
    And do a new check after some training

---

# Week 2 - Optimizing Algorithms

## Batch / Mini-batch / Stochastic gradient descent:

Batch = Calculate gradient every whole training set , e.g. 1000 sample

Mini-batch = Calculate it for every subset of the whole training set, e.g. every 500 sample, i.e. 2 batches here

Stochastic = Calculate it for every SINGLE sample

Batch method only update after whole dataset, so the training speed can become slow.

Stochastic update the parameters for every sample, so it cannot make the most of vectorization, i.e. the calculation is slower, and the worst case is that it never reaches global minima.

As a result, mini-batch takes the good part from both of them.

Normally, the `batch size` is set to a number of $2^{x}$ (e.g. 64, 128…etc) to fit the CPU/GPU memory

![Screen Shot 2022-09-13 at 16.59.09.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-13_at_16.59.09.png)

![Screen Shot 2022-09-13 at 17.03.11.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-13_at_17.03.11.png)

## Fundamental concept - exponentially weighted averages:

This part is a fundamental knowledge before learning other optimization algorithms, apart from Gradient Descent.

To draw an average line, we can use exponentially weighted averages here.

Here is a simplified calculation, without bias correction yet.

e.g. the point at `V1` is calculated by a weighted sum of previous `V0` and current real value.

So the weight here can control how many past records take influence to the current calculating value.

(in the screenshot, green line = large beta, it appears to shift right side because it is “slower to change”)

![Screen Shot 2022-09-13 at 17.05.26.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-13_at_17.05.26.png)

![Screen Shot 2022-09-13 at 17.07.22.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-13_at_17.07.22.png)

Note - the above calculation will have a very small value in the early records.
So we sometimes apply a `Bias correction`

The `V` value is calculated in the same way but divided by $(1-β^{t})$ at last. So the calculated average value is more close to where they should be.

Also when `t` is larger (i.e. in a later point of data), this value become ~ 1 so it takes very less effect later, which is what we want.

Just keep in mind that if the iteration is large, most people will just ignore this bias correction.

![Screen Shot 2022-09-13 at 17.35.49.png](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167/Screen_Shot_2022-09-13_at_17.35.49.png)

---

# Week 3