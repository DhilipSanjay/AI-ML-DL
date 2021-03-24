# Week 4 - Using Real World Images

### Image Generator
- In real world, we will got get images separated into training and validation datasets. We have to do it manually.
- Instead, we can point the image generator at the training directory and all the images in each directory will be loaded and labelled automatically.
    - **Example Directory structure**: 
        - Training Directory
            - Horse
            - Human
        - Validation Directory
            - Horse
            - Human
            
- `rescale` -> To normalize the data.
- `flow_from_directory` - Point it to the directory that contains the sub directory that contains your images. Names of sub directory => labels.
- `target_size` - Images will be resized to the specified size as they are loaded. Experiment with different sizes.
- `batch_size` - Images will be loaded for training and validation in batches. This will be more efficient than doing it one by one. Experiment with different batch sizes.
- `class_mode` - Binary classifier.

### Training ConvNet to use complex images
- Higher complexity and size of the images => More convolution Pooling layers.
- For colored images, we need 3 bytes for pixels (RGB) => Last argument in `input_shape` will be 3.
- `sigmoid` activation function - great for binary classification => Where one class will tend towards zero and other class towards one.
    - We can also use `softmax` function with two neurons (but it's not efficient).
- Loss function - `binary_crossentropy`
- Optimizer function - `RMSprop(lr=0.001)` => We can specify the learning rate in this RMS prop function.
- `steps_per_epoch` - Should be based on `batch_size` and the total number of images.
    - **Example 1:** 
        - Number of images = 1024
        - Batch size = 128
        - Step per epoch should be 8 (1024/128 = 8)
    - **Example 2:**
        - Number of images = 512
        - Batch size = 32
        - Step per epoch should be 16 (512/32 = 16)

### Exploring the impact of compressing images
- Here the images' dimesions are reduced from **300x300** to **150x150**. 
- And also fourth and fifth convolution pooling layers are removed.
- If you get an accuracy of `1.000` on the test data, then it is a sign that you are overfitting.
- Notice where your model fails and try to add such images in the training dataset.

### TensorFlow References
- [Tensorflow - RMSprop](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop)

---

### Colab
- [Horses or Human classifier](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb)
    - Using the **RMSprop optimization algorithm** is preferable to **stochastic gradient descent** (SGD), because RMSprop automates learning-rate tuning for us. (Other optimizers, such as **Adam** and **Adagrad**, also automatically adapt the learning rate during training, and would work equally well here.
    - We go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being **"activated"**; most are set to zero. This is called **"sparsity"**. Representation sparsity is a key feature of deep learning.
- [Horses or Humans with Validation](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb)
- [Horses or Humans with Compacting of Images](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%204%20-%20Notebook.ipynb)

### Reference Links
- [Binary Clasification by Andrew NG](https://www.youtube.com/watch?v=eqEc66RFY0I)
- [Mini-batch gradient descent](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [Cross Entropy Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
    - **Multi-Class Classification** - One-of-many classification. Each sample can belong to ONE of the classes.
    - **Multi-Label Classification** - Each sample can belong to more than one class.
    - **Types of losses**
        - Cross-Entropy Loss
        - Categorical Cross-Entropy loss
        - Binary Cross-Entropy Loss
        - Focal Loss

### Exercises
- [Exercise 4 - Happy or Sad Classifier](Exercise_4_Happy_or_Sad_Classifier_.ipynb)
    - Remember to calculate `steps_per_epoch`.

---