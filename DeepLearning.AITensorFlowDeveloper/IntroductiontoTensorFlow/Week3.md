# Week 3 - Enhancing vision with Convolutional Neural Network

### Convolutional Neural Network
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

### Tensorflow References
- [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
- [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D)

### Implementing CNNs
- Along with the previously added layers, we'll include the CNN layer.
- `input_shape=(28, 28, 1)` - The third argument denotes that we are using **single byte** for color depth (Grayscale).
- `filters` - Mandatory Conv2D parameter is the **numbers of filters** that convolutional layers will learn from. It is an integer value and also determines the number of output filters in the convolution.

### Implementing Max Pooling
- We are going to take the maximum value.
- Pool size = 2x2 (Totally four pixels)
- We can also add another Convolution and Pooling layer.
- `model.summary()` -> shows the journey of the image through convolutions.
- One pixel margin all around the margin cannot be used -> because they won't neighbours => If the filter is 3x3, then the Output of the convolution will be 2 pixel smaller on x and two pixels smaller on y. 
- If the pool size is 2x2, x and y pixels will be halved. (Totally quartered)
- On flattening the output of the Convolutions, we get more elements than that of DNN.
- **Layers API**
    - With the layers API, we can see the output at each layer.

---

### Colab
- [Improving Computer Vision Accuracy using Convolutions](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)
    - `model.summary()` to view the summary of your model.
- [How convolutions work ](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb)
    - Don't forget to change the Runtime Type to **GPU**.
    - Try out this filter (for edge detection):
    ```py
    filter = [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]]
    ```

## Reference Links
- [Image Filtering tutorial](https://lodev.org/cgtutor/filtering.html)

## Exercises
- [Improving Computer Vision Accuracy using Convolutions - Exercises](Improving_Computer_Vision_Accuracy_using_Convolutions_Exercises.ipynb)
- [Exercise 3 -Improve MNIST with convolutions - Solution](Exercise_3_Improve_MNIST_with_convolutions_Solution.ipynb)

---