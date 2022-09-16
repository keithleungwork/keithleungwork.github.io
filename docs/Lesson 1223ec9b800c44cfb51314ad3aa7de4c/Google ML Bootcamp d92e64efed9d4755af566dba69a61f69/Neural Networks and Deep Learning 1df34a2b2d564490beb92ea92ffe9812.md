# Neural Networks and Deep Learning

> Some images and text are from [https://www.deeplearning.ai/](https://www.deeplearning.ai/)
> 

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
| 1 | 14 Aug |
| 2 | 21 Aug |
| 3 | 28 Aug |
| 4 | 4 Sep |

# Goal of this course

- Build neural network, train it with data
- Example  - recognize cat

---

# Week 1 - Introduction to Deep Learning

ReLU function

- rectified linear unit
- rectify = max( X, 0 ) to ensure always ≥ 0

Densely connected

- e.g. All input features are connected to all nodes in the next layer, like below
    
    ![Screen Shot 2022-08-09 at 13.27.16.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-09_at_13.27.16.png)
    

---

Structure tabular data → standard neural network architecture

Computer vision → Convolutional neural network CNNs

Sequence data (audio, text, time-series…etc) → recurrent neural network RNNs

![Screen Shot 2022-08-09 at 14.10.42.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-09_at_14.10.42.png)

Thanks to DL, computer can understand more unstructured data than the past.

(I.e. DL is very good at UNSTRUCTURED data)

---

One of the reasons WHY DL become more popular in recent years

- Thanks to lifestyle technology advance, e.g. mobile, camera, sensors…etc
- i.e. we got much more data
- (Just a conceptual chart) we can see, 
traditional ML algorithm(red) doesn’t know how to handle large size of data.
While Neural Network(esp. larger size) are benefited from the large size of data.

![Screen Shot 2022-08-09 at 14.22.24.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-09_at_14.22.24.png)

- So it is a simple chart to understand : if less data, we should go for simpler algorithm, or smaller NNs

Another reason - Faster Computation

- Switching from Sigmoid func → ReLU func, it enables the training to be way more faster
(I don’t get it though. Maybe checkout the gradient descent formula, that relates to activation func)

![Screen Shot 2022-08-09 at 14.28.57.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-09_at_14.28.57.png)

So this cycle is faster

![Screen Shot 2022-08-09 at 14.31.26.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-09_at_14.31.26.png)

---

---

# Week 2 - Neural network basics

## Section 1 - Logistic Regression as a neural network

### **Binary Classification**

- can use logistic regression
- Equation looks like:
    
    ![Screen Shot 2022-08-16 at 12.36.52.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_12.36.52.png)
    

Notation Example - classification of an image → cat or not

- Let’s say image is 64 x 64 pixel, with RGB 3 channel
- input X - a feature vector with dimension $n_x$ of  (64x64x3)
    - e.g. $x_1 = \begin{bmatrix} 23 \\ 234 \\ 2 \\ ... \end{bmatrix}$ then, whole $X = \begin{bmatrix} x_1, x_2, ..., x_m \end{bmatrix}$, with the shape of $( n_x, m )$
    - Andrew claimed, put 1 entity as a column instead of a row that will be easier for later implementation
- output Y - 1 or 0

Its equation is very similar to linear regression, EXCEPT we want the output to be a probability(i.e. between 0 to 1). So we add a sigmoid function to the result of linear regression equation.

i.e. y = sigmoid(wx + b)

where w is an n dimension vector, b is a real number

![Screen Shot 2022-08-16 at 12.23.20.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_12.23.20.png)

Alternative notation: 

Sometimes people notate the equation slightly differently. They put `w` and `b` into the same parameters theta. The first element (theta zero) is actually `b`, and then `x` become a `n+1` vector, with the x zero equal to 1.

![Screen Shot 2022-08-16 at 12.27.57.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_12.27.57.png)

### Cost function:

For Logistic Regression, we cannot use the general square root error equation.
Reason is there will not be easy to find global optima, but a lot of local optima.

Instead, we would use some similar equations like this effect

![Screen Shot 2022-08-16 at 12.44.11.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_12.44.11.png)

Diff with `Loss function, Error function`???

- Loss/Error Function is how we measure the loss of ONE SINGLE training sample
- While Cost Function, is how we measure the ENTIRE training set
- J(w, b) is the cost function, while L(y, y) is the Loss function
    
    ![Screen Shot 2022-08-16 at 13.07.38.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_13.07.38.png)
    

### Gradient descent:

It is the process to minimize the cost function to the global optima, by updating parameters `w` and `b` with the step calculated by learning rate & its slope

![Screen Shot 2022-08-16 at 13.17.18.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_13.17.18.png)

### Computation Graph:

- Useful to understand the flow and the output we want to optimize.
- During a `forward propagation`, we calculate the `Cost Function`
- During a `backward propagation`, we yield the `derivative`

For example,

Let’s say we want to see how `J` changes when `a` changes.

- `J` changes when `v` changes → $\frac{dJ}{dv}$
- `v` changes when `a` changes → $\frac{dv}{da}$
- So, $\frac{dJ}{da} = \frac{dJ}{dv} . \frac{dv}{da}$ , which calls the `chain derivatives`
    
    ![Screen Shot 2022-08-16 at 14.29.50.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_14.29.50.png)
    
- Other parameter `b` or `c` are calculated in the same way, with 1 more layer though.
    - $\frac{dJ}{db} = \frac{dJ}{dv}.\frac{dv}{du}.\frac{du}{db}$ , and `c` is in the same way
- In the `Back Propagation` phrase, those (partial) derivatives are calculated, then each parameters can be updated, by a value depending on `derivatives` and `learning rate`
    
    ![Screen Shot 2022-08-16 at 15.10.10.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_15.10.10.png)
    

Note that the output is a `Loss Function` here (for ONE sample)
By Backward propagation, we can calculate $\frac{dL}{dz}$ and reuse it, to calculate the derivatives of `w1`, `w2`, `b`

![Screen Shot 2022-08-16 at 15.27.16.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-16_at_15.27.16.png)

Similarly, for the `Cost Function`, we just need to take average of the sum of `Loss Function` above (also same for other derivatives dw1, dw2, db), i.e.

$$
\frac{dJ}{dw^1} =
\frac{1}{m}
\displaystyle\sum_{i=1}^m
\frac{dL}{dw^1}
$$

Side notes:

In the above calculation, we may use a for loop to iterate `m` samples, and another nested for loop to iterate `n` parameters (e.g. w1 ~ wn). But in deep learning, we need to avoid for loop as much as we can. So a technique called `Vectorization` is important to get rid of most for loop.

---

## Section 2 - Python and Vectorization

In short, avoid using For loop to process list/matrix whenever possible.

For example 

- use numpy matrix operation (e.g. sum…etc) for list/matrix
- use Python Broadcasting feature to do calculation
    - e.g.
    
    ```python
    A = [ 1, 2, 3 ] + 100
    # A: [ 101, 102, 103 ]
    ```
    

Example of vectorizing the basic equation $z = w^{T}X + b$  in a single node

- $w = 
\begin{bmatrix}
w_1 \\
w_2 \\
...
\end{bmatrix}$ where each `w` is for multiplying each `x`
- $X = 
\begin{bmatrix}
x_1 \\
x_2 \\
...
\end{bmatrix}$
where each x refer to each `Features`
- So, $w^{T}X$ is a scalar value
    
    

Example of avoiding for loop in the calculation of `derivatives in Logistic regression`

![Screen Shot 2022-08-20 at 23.06.50.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-20_at_23.06.50.png)

- 1st loop, looping `m` training example
- 2nd loop, looping x1, x2, x3…xn features in 1 sample

Then the above 2 loops can be converted into Vectorization as below

![Screen Shot 2022-08-20 at 23.10.08.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-20_at_23.10.08.png)

> Cautions! When dealing with vectorization operation, always avoid using rank 1 array.
But you should explicitly create the proper dimension even it is with the shape `1`. As shown below
> 

![Screen Shot 2022-08-20 at 23.14.44.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-20_at_23.14.44.png)

### Numpy exercise:

math.exp() vs numpy.exp()

- We barely use math package in DL, because always handle matrix
- numpy automatically support input as real number, or vector, matrix…etc

np.reshape(-1)

Normalization is important

- it allow ML/DL gradient descent to converge faster
- e.g. via `np.linalg.norm(x, axis=1, keepdims=True)`

---

# Week 3 - Shallow Neural Networks

A NN with one hidden layer is generally called `2 Layer NN`

Because the output layer is counted but the input layer is not.

And input layer `X` sometimes is referred to `a[0]` while other layers i are `a[i]`

![Screen Shot 2022-08-25 at 13.27.37.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_13.27.37.png)

In a logistic regression simple NN, each neuron (node) can be presented in 2 parts like this:

![Screen Shot 2022-08-25 at 13.29.35.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_13.29.35.png)

## Vectorization of one layer

From previous week2 lecture we learnt Vectorization of ONE NODE

And here is the implementation on the one hidden layer, with multiple nodes

> superscript square bracket `[1]` = layer 1
subscript `1` = node 1 in this layer
> 

By recalling the basic vectorization in previous week2.

- $z^{[1]}_1$  is the value in node 1 in layer 1
- $w^{[1]}_1 = 
\begin{bmatrix}
w_1 \\ w_2 \\ w_3 \\ ...
\end{bmatrix}$ , and   $x = 
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ ...
\end{bmatrix}$ , and  $b^{[1]}_1$  is a scalar
- So, $z^{[1]}_1 = w^{[1]T}_1 x + b^{[1]}_1$  is a scalar value

From above image, each line of equation in the top right corner is for each nodes.

By applying vectorization to the whole layer, e.g. layer 1 

![Screen Shot 2022-08-25 at 13.31.33.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_13.31.33.png)

$z^{[1]} = W^{[1]} x + b^{[1]} = 
\begin{bmatrix}
w^{[1]T}_1 \\
w^{[1]T}_2 \\
...
\end{bmatrix}

 \begin{bmatrix}
x_1 \\
x_2 \\
...
\end{bmatrix}

+ \begin{bmatrix}
b^{[1]}_1 \\
b^{[1]}_2 \\
...
\end{bmatrix}

\\$
$=
\overbrace{
\begin{bmatrix}
w^{[1]}_{1,1} & w^{[1]}_{1,2} & ... \\
w^{[1]}_{2,1} & w^{[1]}_{2,2} & ... \\
... & ... & ...
\end{bmatrix}
}^{W^{[1]}}
 \begin{bmatrix}
x_1 \\
x_2 \\
...
\end{bmatrix}
+
 \begin{bmatrix}
b^{[1]}_1 \\
b^{[1]}_2 \\
...
\end{bmatrix}$
  , where $w^{[1]}_{1,2}$  ,for example, is `w`  in `node 1` for $x_2$

$=

\begin{bmatrix}
w^{[1]}_{1,1}x_1 + w^{[1]}_{1,2}x_2 + ... + b^{[1]}_1 \\
w^{[1]}_{2,1}x_1 + w^{[1]}_{2,2}x_2 + ... + b^{[1]}_2 \\
...
\end{bmatrix}$

$= \begin{bmatrix}
z^{[1]}_1 \\
z^{[1]}_2 \\
...
\end{bmatrix}$

For the shape of above equation:

z (4, 1) = W (4, 3) x (3, 1) + b (4, 1)

where `4` is nodes size, `3`is feature size

Thus, the output of `layer 1` & `layer 2` are as below:

![Screen Shot 2022-08-25 at 16.40.38.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_16.40.38.png)

## Vectorizing across the whole dataset(`X`):

From the final few equations above, for layer 1 and layer 2, we could think of a for loop easier like left side below. And then convert into vectorization as right side:

- $z^{[1](i)}$  with the shape (4,1)
    - $Z^{[1]}$ with the shape (4, m) , where `m` is the size of dataset `X`
- $W^{[1]}$ with the shape (4,3)
- X with the shape (3, m)
- $b^{[1]}$ with the shape (4,1)

![Screen Shot 2022-08-25 at 16.55.15.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_16.55.15.png)

## Activation function

Sigmoid:

For Andrew, he claimed he barely used `sigmoid` EXCEPT when the output is a binary classification. i.e. 0 ≤ y ≤ 1. Then the `sigmoid` is used in the output layer

***Note***: so the activation function CAN be different between layers

Tanh:

In short, because it goes through the point (0,0), that the mean is near zero, easier for calculation. It is ALMOST alway better than `sigmoid`, EXCEPT when the output is in the range of (0,1)

ReLU:

However, both `sigmoid` & `tanh` also have a downside. When the z (input of activation function)  is very large / very small, its derivative(slope) is very small. 
i.e. the gradient descent is small and backward update will be very slow. In such cases, other activation such as `ReLU` can perform better. (rectifier linear unit)

Or `leaky ReLU`...

![Screen Shot 2022-08-25 at 17.22.07.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_17.22.07.png)

In conclusion, choose the activation function depends on the output. If not sure which activation function to use, try to use `ReLU` or `tanh` first. And the choice of these activation function would bring impact to the training speed.

### Why need nonlinear activation function ?

Put it in short, if use a linear activation function and after factorization of the equation, the final equation of the whole network would be still like $y = w^{'}x + b^{'}$

i.e. no matter how many hidden layers, it is similar to a single layer.

So `linear activation function` is barely used except in output layer for some rare cases.

## Gradient descent of Neural Network

In the course, there are lots of derivatives calculation.

Here is the summary of the equations, for the example in the slides above, with 2 layer NN.

![Screen Shot 2022-08-25 at 18.49.30.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_18.49.30.png)

## Random Initialization

***Avoid initialize weights to zero***

Otherwise all unit in the same layer will be the same no matter after any times of iteration and update. i.e. meaningless how many units you have

Also units are not updated correctly or even not updated

> Seems, it is called `symmetry breaking problem`
> 

> Note: `b` is okay to be zero
> 

![Screen Shot 2022-08-25 at 18.54.00.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_18.54.00.png)

***Correct way - random initialization***

> Note: Below is a simple version. A more practical way, please refer to the content in [course 2 here](Improving%20Deep%20Neural%20Networks%20Hyperparameter%20Tuni%20b0fd7167ddb34984a40b0dd550166167.md)
> 

Weights should be initialized randomly AND to be some small  numbers ( Show as the `*0.01` in the below equation, note that it could be other values, 0.001, …etc)

Why ?

- if `w` are too large → `z` is too large → `slope of activation` ~= 0
- then the backward propagation update is very very slow

![Screen Shot 2022-08-25 at 18.57.23.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-25_at_18.57.23.png)

## Assignment

Some shapes of variable used in the functions

```python
"""
X shape: (2, 400) a.k.a (n_x, m)
Y: (1, 400)

- n_x: the size of the input layer
- n_h: the size of the hidden layer (**set this to 4, only for this Exercise 2**) 
- n_y: the size of the output layer
"""

(n_x, n_h, n_y) = layer_sizes(t_X, t_Y)
# 2, 4, 1
parameters = initialize_parameters(n_x, n_h, n_y)
"""
parameters = {"W1": W1, (n_h, n_x)
              "b1": b1, (n_h, 1)
              "W2": W2, (n_y, n_h)
              "b2": b2} (n_y, 1)
"""

A2, cache = forward_propagation(t_X, parameters)
"""
A2 : (1, m)

Z1:  (n_h, m)
A1:  (n_h, m)
Z2:  (1, m)
A2:  (1, m)
"""

cost = compute_cost(A2, t_Y)

X : (2, m)
Y : (1, m)

grads = backward_propagation(parameters, cache, t_X, t_Y)
"""
grads = {"dW1": dW1, (n_h, 1)
         "db1": db1, (1,)
         "dW2": dW2, (1, n_h)
         "db2": db2} (1,)
"""

parameters = update_parameters(parameters, grads)

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False) -> parameters
```

![Screen Shot 2022-08-26 at 22.13.52.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-26_at_22.13.52.png)

![Screen Shot 2022-08-26 at 22.21.13.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-26_at_22.21.13.png)

![Screen Shot 2022-08-26 at 22.31.58.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-26_at_22.31.58.png)

![Untitled](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Untitled.png)

---

# Week 4 - Deep Neural Network

From previous weeks, we learnt the basic of how neuron / each node works in a shallow NN. And we learnt to calculate:

- In forward propagation, vectorization with Z = WX + B and the activation function A = G(Z)
- In back propagation, vectorization of several derivatives (dA, dZ, dW, dB) and how to update parameters

In this short week 4 content, we just apply above knowledge to a deeper NNs with more layers. And we can observe a general formula

- $Z^{[l]} = W^{[l]} A^{[l-1]} + B^{[l]}$, where $A^{[0]}$ is equal to X
- $A^{[l]} = g^{[l]}(Z^{[l]})$
- And the formula of back propagation is shown in the screenshot

Note: It is worth caching the `Z` & `A` value during calculating each node, that will be reused in back propagation.

![Screen Shot 2022-08-29 at 12.46.46.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_12.46.46.png)

![Screen Shot 2022-08-29 at 12.46.55.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_12.46.55.png)

### Why deeper layers?

Andrew used an example of XOR results of x1, x2, x3, ….xn to illustrate. So it is said that deeper layers is better than more units with shallow layers

### What is Hyperparameters?

They are another kind of parameters, that affect how real parameters (W, B) will be calculated.

e.g. Learning rate, iterations, layer size, nodes size, activation function choice, …etc

### Quiz this week:

- Confusion point:
    - We don’t calculate dA^l from Z^l, why?
        - $\frac{dJ} {dA^{[l]}} = 
        \frac{dJ} {dZ^{[l+1]}} . \frac{dZ^{[l+1]}} {dA^{[l]}} = 
        \frac{dJ} {dZ^{[l+1]}} . 
        \frac{d} {dA^{[l]}} (W^{[l+1]}A^{[l]} + b^{[l+1]})
        
        \\ \text{ } \\
        = dZ^{[l+1]}  W^{[l+1]}$
        

### Assignment part 1 of 2 :

Here's an outline of the steps in this assignment:

![Screen Shot 2022-08-29 at 15.50.01.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_15.50.01.png)

- Initialize the parameters for a two-layer network and for an L-layer neural network
- Implement the forward propagation module (shown in purple in the figure below)
    - Complete the LINEAR part of a layer's forward propagation step (resulting in Z^[l]).
    - The ACTIVATION function is provided for you (relu/sigmoid)
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
    - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer L). This gives you a new L_model_forward function.
- Compute the loss
- Implement the backward propagation module (denoted in red in the figure below)
    - Complete the LINEAR part of a layer's backward propagation step
    - The gradient of the ACTIVATE function is provided for you(relu_backward/sigmoid_backward)
    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function
    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
- Finally, update the parameters

The Network architecture we will implement is:

![Screen Shot 2022-08-29 at 16.59.47.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_16.59.47.png)

Variable shape tracking:

```python
parameters = initialize_parameters(3,2,1)
"""
W1 -- weight matrix of shape (n_h, n_x)
b1 -- bias vector of shape (n_h, 1)
W2 -- weight matrix of shape (n_y, n_h)
b2 -- bias vector of shape (n_y, 1)
"""

parameters = initialize_parameters_deep([5,4,3])
""" A dict containing all layers param : 
{
	W1 -- weight matrix of shape (n_h, n_x)
	b1 -- bias vector of shape (n_h, 1)
	W2 -- weight matrix of shape (n_y, n_h)
	b2 -- bias vector of shape (n_y, 1)
	...
}
"""

# Pure Z = WX + b
t_Z, t_linear_cache = linear_forward(t_A, t_W, t_b)
"""
t_A : (n_unit_prev, m)
t_W : (n_unit_now, n_unit_prev)
t_b : (n_unit, 1)

t_Z : (n_unit_now, m)
t_linear_cache : (t_A, t_W, t_b)
"""

# activation_cache contains Z
A, activation_cache = sigmoid(Z)
A, activation_cache = relu(Z)

A, cache = linear_activation_forward(t_A_prev, t_W, t_b, activation = "sigmoid") # or relu
"""
A = (n_unit_now, m)
cache = (linear_cache, activation_cache)
			= (
					(t_A_prev, t_W, t_b),
					Z
				)
"""

# t_parameters from initialize_parameters_deep
t_AL, t_caches = L_model_forward(t_X, t_parameters)
"""
AL : (1, m) # because last layer unit = 1
caches = [
	(linear_cache, activation_cache) # layer 1 cache (relu layer)
	( # 2nd layer here
		(A1, W2, b2),
		Z2
	)
	...
	(linear_cache, activation_cache) # layer L cache (sigmoid output)
]
"""

t_cost = compute_cost(t_AL, t_Y)
# cost : float

# dz: (cur_unit_size, m), linear_cache = (A_prev, W, b)
dA_prev, dW, db = linear_backward(t_dZ, t_linear_cache)
t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation = "sigmoid") # or relu
"""
dA_prev: (prev_unit_size, m)
dW:  (current_unit_size, prev_unit_size)
db:  (current_unit_size ,1) 
"""

# AL is last layer output, i.e. y_hat
grads = L_model_backward(AL, Y_gt, caches)
"""
grads: {
	dA0: (layer_0_unit_size, m), i.e. (X_feature_size, m)
	dW1: (layer_1_unit_size, layer_0_unit_size)
	db1: (layer_1_unit_size, 1)
	...
}
"""

t_parameters = update_parameters(t_parameters, grads, 0.1)
""" A dict containing all layers param : 
{
	W1 -- weight matrix of shape (n_h, n_x)
	b1 -- bias vector of shape (n_h, 1)
	W2 -- weight matrix of shape (n_y, n_h)
	b2 -- bias vector of shape (n_y, 1)
	...
}
"""
```

**Memo:**

Implementation of cost function

```python
cost = (
	np.dot(Y, np.log(AL.T)) 
	+ np.dot(1-Y, np.log(1-AL.T))
) / -m
```

![Screen Shot 2022-08-29 at 17.25.33.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_17.25.33.png)

Implementation of derivatives

```python
dW = np.dot(dZ, A_prev.T) /m # (current_unit_size, prev_unit_size)
db = np.sum(dZ, axis=1, keepdims=True) /m # (current_unit_size ,1)
dA_prev = np.dot(W.T, dZ) # (prev_unit_size, m)
```

![Screen Shot 2022-08-29 at 17.39.26.png](Neural%20Networks%20and%20Deep%20Learning%201df34a2b2d564490beb92ea92ffe9812/Screen_Shot_2022-08-29_at_17.39.26.png)

Implementation of cost derivative

```python
dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
```

$$
\frac{dJ} {dA^{[L]}} =
- (
\frac{Y} {A^{[L]}}
- \frac{1-Y} {1-A^{[L]}} )
$$

### Assignment part 2 of 2:

Shape memo:

```python
# train_x's shape: (12288, 209)
# test_x's shape: (12288, 50)
# train_y shape: (1, 209)
# test_y shape: (1, 50)

# Reshape and standardize procedure
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# 1. 2-layer nn
# LINEAR -> RELU -> LINEAR -> SIGMOID.

"""From prev assigment
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
"""

parameters, costs = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2, print_cost=False)
"""
parameters : final param of W1, b1, W2, b2
costs: list of each calculated cost (float)
"""
```

Code in the 2-layer func

```python
parameters = initialize_parameters(n_x,n_h,n_y)

# X: (n_x, m)
# W1: (n_h, n_x)
# b1: (n_h, 1)
A1, cache1 = linear_activation_forward(X, W1, b1, activation = "relu") # or relu
# A1: (n_h, m)
# W2: (1, n_h)
# b2: (1, 1)
A2, cache2 = linear_activation_forward(A1, W2, b2, activation = "sigmoid") # or relu
# A2: (1, m)

cost = compute_cost(A2, Y)

dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation = "sigmoid") # or relu
dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu") # or relu
parameters = update_parameters(parameters, grads, learning_rate)
```