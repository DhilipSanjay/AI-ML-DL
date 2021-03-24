# Week 2 - Introduction to Computer Vision

### Computer Vision
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


### Callbacks
- To stop the training after reaching a particular level of accuracy or if the error falls below a particular level.
- A **list of callback functions** can be applied during the training of our model.

---

### Colab
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

### Reference Links
- [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [Responsible AI practices](https://ai.google/responsibilities/responsible-ai-practices/)
    

### Problems faced
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

### Exercises
- [Exercise 2 - Handwriting recognition](Exercise2_Handwriting_Recognition_Solution.ipynb)

---