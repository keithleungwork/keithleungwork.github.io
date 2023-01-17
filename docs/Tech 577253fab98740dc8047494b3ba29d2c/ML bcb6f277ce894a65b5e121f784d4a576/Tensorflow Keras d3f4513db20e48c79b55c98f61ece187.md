# Tensorflow / Keras

---

## Data

### Split image data into train, dev set with `image_dataset_from_directory`

- ref - [https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)

Usage sample :

```python
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=42)
```

### Using `<dataset>.prefetch` to prevent memory bottleneck

- Memory bottleneck that can occur when reading from disk. This method sets aside some data and keeps it ready for when it's needed, by creating a source dataset from your input data, applying a transformation to preprocess it, then iterating over the dataset one element at a time. Because the iteration is streaming, the data doesn't need to fit into memory.
- You can set the number of elements to prefetch manually, or you can use `tf.data.experimental.AUTOTUNE` to choose the parameters automatically. Autotune prompts `tf.data` to tune that value dynamically at runtime, by tracking the time spent in each operation and feeding those times into an optimization algorithm. The optimization algorithm tries to find the best allocation of its CPU budget across all tunable operations.

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
```

---

## Data Augmentation (Vision)

Keras' Sequential API offers a straightforward method for these kinds of data augmentations, with built-in, customizable preprocessing layers. These layers are saved with the rest of your model and can be re-used later. Ahh, so convenient!

- [https://www.tensorflow.org/tutorials/images/data_augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

```python
# A function for data augmentation
def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    data_augmentation = tf.keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    
    return data_augmentation

## usage
data_augmentation = data_augmenter()
# simple test: augmented_image = data_augmentation( [first_image] )
# some input layers
inputs = tf.keras.Input(shape=input_shape)
# apply data augmentation to the inputs
x = data_augmentation(inputs)
# some other layers.....
```

---

## Model

### Compile model

- When the model is created, you can compile it for training with an optimizer and loss of your choice.
- When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile)

```python
created_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
# or
model2.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### Train a model

```python
history = happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)

# below is a common code to visualize history

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

### Evaluate a model

```python
happy_model.evaluate(X_test, Y_test)
```

---

## 

---

## Manually Create Neural Network

Prerequisite Knowledge:

- TF Keras Layers - [https://www.tensorflow.org/api_docs/python/tf/keras/layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
- Sequential API - [https://www.tensorflow.org/guide/keras/sequential_model](https://www.tensorflow.org/guide/keras/sequential_model)
    - You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.
    - Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.
    - If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!
- Or, Functional API - [https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
    - more flexible than the `tf.keras.Sequential` API
    - Can handle NON-linear topology, such as Skip connection(e.g. ResNet)
    

### Common Layers:

- [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): pad the image with zero
- [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Pooling layer
- [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): CNN layer
- [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
    - Importantly, batch normalization works differently during training and during inference.
- [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
- [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)
- Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with N neurons and activation (e.g. sigmoid).

### Build the model architecture:

- With sequential API:

```python
model = tf.keras.Sequential([
#... all layers definition
])

```

- With functional API:

```python
# Define Input layer
input_img = tf.keras.Input(shape=input_shape)
# Define all hidden layers
# e.g. 
# First component of main path
X = Conv2D(
	filters = 10, kernel_size = 1, strides = (1,1),
	padding = 'valid', kernel_initializer = initializer(seed=0)
)(X)
X = BatchNormalization(axis = 3)(X, training = training) # Default axis
X = Activation('relu')(X)
# ...
outputs = tf.keras.layers.Dense(32)(X)

# Create model
model = tf.keras.Model(inputs=input_img, outputs=outputs)
```