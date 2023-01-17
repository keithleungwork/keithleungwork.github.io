# Sequence Models

---

## Cite

As in copyright:

> [DeepLearning.AI](https://www.deeplearning.ai/) makes these slides available for educational purposes. You may not use or distribute these slides for commercial purposes. You may make copies of these slides and use or distribute them for educational purposes as long as you cite [DeepLearning.AI](https://www.deeplearning.ai/)  as the source of the slides.
> 
- Slides here are from [DeepLearning.AI](https://www.deeplearning.ai/)

---

## Week 1

### Assignment 1

**RNN Architecture:**

- Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory.”
- The example of RNN with len(input) = len(output)
    
    ![Screen Shot 2022-12-27 at 13.39.03.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_13.39.03.png)
    
    - Each input $x^{(t)}$can be an one-hot vector, or just a value. 
    e.g. A language with a 5000-word vocabulary could be one-hot encoded into a vector that has 5000 units. So $x^{(t)}$ shape = (5000,)
    - The activation a⟨t⟩ that is passed to the RNN from one time step to another is called a `hidden state`
    

**RNN Cell:**

- Think of the recurrent neural network as the repeated use of a single cell
- The following figure describes the operations for a single time step of an RNN cell:
    
    ![Screen Shot 2022-12-27 at 14.44.17.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_14.44.17.png)
    

**LSTM Cell:**

- An LSTM is similar to an RNN in that they both use hidden states to pass along information, but an LSTM also uses a cell state, which is like a long-term memory, to help deal with the issue of vanishing gradients
- An LSTM cell consists of a cell state, or long-term memory, a hidden state, or short-term memory
- Here is the cell detail:
    
    ![Screen Shot 2022-12-27 at 15.30.39.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_15.30.39.png)
    
- The simple explanation of the flow through:
    - Similar to simple RNN, `a(t-1)` and `x(t)` are the inputs (near the bottom left)
    They are used to calculate:
        - Forget Gate - A “mask” vector, used for another input `c(t-1)` calculation, between 0~1
        - Update Gate - A “mask” vector, that decides if the Candidate values `𝐜̃ ⟨𝑡⟩` can pass into the hidden state `c(t)` , between 0~1
        - Candidate value - the value calculated from previous activation and current input, between -1~1
        - Output Gate - A “mask” vector, decides what values are passed into Output y, and next activation
    - `c(t-1)` is another new input here (left side)
        - With the Forget Gate from above, values are decided to whether pass in or not
        - Then add with the final candidate value, it become the next `c(t)`

- About the `Forget Gate`:
    
    ![Screen Shot 2022-12-27 at 15.34.50.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_15.34.50.png)
    
- About the `Candidate Value`
    
    ![Screen Shot 2022-12-27 at 15.46.58.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_15.46.58.png)
    
- `Update gate` :
    
    ![Screen Shot 2022-12-27 at 15.57.51.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_15.57.51.png)
    
- `Output gate`:
    
    ![Screen Shot 2022-12-27 at 16.00.36.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_16.00.36.png)
    
- At last, final `cell state`
    
    ![Screen Shot 2022-12-27 at 15.58.30.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_15.58.30.png)
    
- `Hidden state` for next cell
    
    ![Screen Shot 2022-12-27 at 16.16.41.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_16.16.41.png)
    
- Prediction output
    
    ![Screen Shot 2022-12-27 at 16.17.05.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_16.17.05.png)
    

### Assignment 2:

![Screen Shot 2022-12-27 at 18.00.34.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_18.00.34.png)

```python
x (27, 1)
Wax (100, 27)
Waa (100, 100)
b (100, 1)
a (100, 1)

Wya (27, 100)
by (27, 1)
y (27, 1)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
clip(gradients, maxValue)
def sample(parameters, char_to_ix, seed):
optimize(X, Y, a_prev, parameters, learning_rate = 0.01):

def rnn_forward(X, Y, a_prev, parameters):
    """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
    It returns the loss' value as well as a "cache" storing values to be used in backpropagation."""
    ....
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    """ Performs the backward propagation through time to compute the gradients of the loss with respect
    to the parameters. It returns also all the hidden states."""
    ...
    return gradients, a

def update_parameters(parameters, gradients, learning_rate):
    """ Updates parameters using the Gradient Descent Update Rule."""
    ...
    return parameters

np.random.seed(0)
probs = np.array([0.1, 0.0, 0.7, 0.2])
idx = np.random.choice(range(len(probs)), p = probs)
```

**Generating text:**

- After the language model is trained, we can use it to generate pieces of text
    
    ![Screen Shot 2022-12-27 at 17.59.03.png](Sequence%20Models%20ebfd9f6947f042b7a08adb5c89d456ed/Screen_Shot_2022-12-27_at_17.59.03.png)
    
    - The `sample` process here is to : sampling a random text in the chosen texts, to prevent generating the same text every time.
        - How to: With the output of softmax, we have a vector of probability. Then we choose the words by taking its probability. 
        E.g. word at i-th index = 16%, it might be picked with a chance of 16%