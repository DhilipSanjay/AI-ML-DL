# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

**Date:** 08, February, 2021

**Author:** Dhilip Sanjay S

---

- [Link to course](https://www.coursera.org/learn/introduction-tensorflow)
- [Neural Networks and Deep learning - Youtube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
- [Convolutional Neural Networks - Youtube Playlist](https://youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF)
---

## Week 1 - A New Programming Paradigm
- **Machine Learning Primer**
    - Rules + Data -> Traditional Program -> Answers
    - Answers + Data -> Machine Learning -> Rules
- **Neural Network basics**
    - Input layer
    - Hidden layer
    - Output layer
- **Colab**
    - ['Hello World' in TensorFlow](https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb)
- **Reference Links**
    - [TensorFlow Youtube Channel](https://www.youtube.com/tensorflow)
    - [Visualize Neural Network - using Playground](http://playground.tensorflow.org/)
- **Exercises**
    - [Exercise 1 - House Price Prediction](Exercise_1_House_Prices_Solution.ipynb)

---

## Week 2 - Introduction to Computer Vision
- The idea of fitting x and y relationship is what lets us do things like computer vision.
- Computer vision is the field of having a computer understand and label what is present is an image.
- **Dataset** - Fashion MNIST. 70,000 Images of clothing.
    - 60,000 for training.
    - 10,000 for testing.
- **Grayscale image** -> amount of information is less.
- Why do we use numbers to label the image?
    - Because if we label it in English, we are being biased towards English speakers.
    - But with numerical data, we can refer to it in our appropriate language.
- Activation functions in each layer of the neuron.
    - **Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
    - **Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!
- **Callbacks**
    - To stop the training after reaching a particular level of accuracy or if the error falls below a particular level.
- **Colab**
    - [Computer Vision in TensorFlow](https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)
        - **Sequential**: That defines a SEQUENCE of layers in the neural network
        - **Flatten**: Flatten just takes that square and turns it into a 1 dimensional set. ('flatten' => 28x28 into a 784x1)
        - **Dense**: Adds a layer of neurons
        - Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now. 
        - **Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network
        - **Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!
        - **Note:**
            - By adding more Neurons we have to do more calculations, slowing down the process, but in this case they have a good impact -- we do get more accurate.
            - The number of neurons in the last layer should match the number of classes you are classifying for.
            -  For far more complex data (including color images to be classified as flowers that you'll see in the next lesson), extra layers are often necessary.
            - Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5. Try 30 epochs -- you might see the loss value stops decreasing, and sometimes increases => Due to overfitting.

    - [Implementing Callbacks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb)
- **Reference Links**
    - [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
    - [Responsible AI practices](https://ai.google/responsibilities/responsible-ai-practices/)
    

- **Problems faced:**
    - In callbacks, accuracy can be `acc` also. To check it, print the logs inside the callback:
    ```py
    print(logs)
    ```
    
    - In **Python 3** -> comparison between `None` and `float` is not allowed. But it is allowed in **Python 2**.
    ```py
    # In Callbacks
    # Python 2
    if (logs.get('accuracy') > 0.99):
    # Python 3
    if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
    ```
- **Exercises**
    - [Exercise 2 - Handwriting recognition](Exercise2_Handwriting_Recognition_Solution.ipynb)
---

## Week 3 - Enhancing vision with Convolutional Neural Network
- Deep neural networks - just classifies based on each and every pixels
- Convolutional neural networks - classify based on features.
- 784 pixels -> out of which many pixels are useless. Therefore, condense the image to important features.
- **Convolution:** 
    - For each pixel -> New pixel value will be sum of the corresponding neighbour values multiplied by the corresponding values of filter. (9x9 matrix)
    - Convolutions will change the image such that certain features get emphasized. (depending on the filter matrix)
- **Pooling:**
    - Way of compressing the image.
    - Out of four pixels (current pixel, its neighbours underneath and to the right of it) -> Pick the biggest value and keep just that.
    - This will preserve the features highlighted by the Convolution.
    - Also quarters the size of the image. (16 pixels -> 4 pixels)

- **Tensorflow References**
    - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
    - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)

- **Implementing CNNs**
    - Along with the previously added layers, we'll include the CNN layer.
    - `input_shape=(28, 28, 1)` - The third argument denotes that we are using **single byte** for color depth (Grayscale).

- **Implementing Max Pooling**
    - We are going to take the maximum value.
    - Pool size = 2x2 (Totally four pixels)
    - We can also add another Convolution and Pooling layer.
    - `model.summary()` -> shows the journey of the image through convolutions.
    - One pixel margin all around the margin cannot be used -> because they won't neighbours => If the filter is 3x3, then the Output of the convolution will be 2 pixel smaller on x and two pixels smaller on y. 
    - If the pool size is 2x2, x and y pixels will be halved. (Totally quartered)
    - On flattening the output of the Convolutions, we get more elements than that of DNN.
- **Layers API**
    - With the layers API, we can see the output at each layer.

- **Colab**
    - [Improving Computer Vision Accuracy using Convolutions](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)
        - `model.summary()` to view the summary of your model.
    - [How convolutions work ](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)
        - Don't forget to change the Runtime Type to **GPU**.

- **Exercises**
    - [Improving Computer Vision Accuracy using Convolutions - Exercises](Improving_Computer_Vision_Accuracy_using_Convolutions_Exercises.ipynb)

## Week 4 - Using Real World Images

---

