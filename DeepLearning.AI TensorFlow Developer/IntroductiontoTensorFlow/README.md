# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

**Date:** 08, February, 2021

**Author:** Dhilip Sanjay S

---

- [Link to course](https://www.coursera.org/learn/introduction-tensorflow)
- [Neural Networks and Deep learning - Youtube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
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
    - [Exercise 1 - House Price Prediction](Exercise_1_House_Prices_Question.ipynb)

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
    - [Implementing Callbacks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb)
- **Reference Links**
    - [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
    - [Responsible AI practices](https://ai.google/responsibilities/responsible-ai-practices/)
---

## Week 3 - Enhancing vision with Convolutional Neural Network

---

## Week 4 - Using Real World Images

---

