# Week 2 - Word Embeddings

- When you have 10000 words, is there a better way to represent these rather than using 1-10000.
-Embeddings represent the **semantics** of the word. Words are represented as a vector in n-dimensional space.
- Words related to positive and negative reviews are clustered separately.
- We can also make use of pretrained word embeddings.

### Embeddings to build a classification for IMDB dataset.
- **TensorFlow Data Services (TFDS)** contains many datasets and lots of different categories.
- To determine the version of tensorflow - `print(tf.__version__)`
- Enable eager execution. (By default enabled in TensorFlow 3.x)
- Install `tensorflow datasets` using pip.
- Perform initial word encoding, padding of the both training and testing data (convert them to numpy array)

### Using vectors
- As the neural network trains, it can learn these vectors associating them with the labels to come up with the **embeddings**. ->  the vectors for each word with their associated sentitment.
- Result of the embedding is a 2D array with the **length of the sentence and the embedding dimension**.
- We'll then Flatten it and pass it into the dense neural network.
- Two options for flattening:
```py
tf.keras.layers.Flatten() # Slower but accurate
tf.keras.layers.GlobalAveragePooling1D() # Averages across the vector to flatten it out. Simpler model and hence faster
```
- Trainig sequences are generated based on the word index learned from the testing sentences. It'll have more OOV. But this is a good test.
- We would need to reverse the key value pair in the word index -> key - index and value - word
- Visualize the embeddings by using the tsv files at https://projector.tensorflow.org/

### Analysing the loss
- Loss - confidence in the prediction.
- As the number of accurate predictions increased over time, the confidence per prediction effectively decreased.
- To overcome this problem - **Decrease the vocabulary size**, **reduce the size of the sentences**. (Changing the number of embeddings has very little impact.)
- Plotting accuracy and loss:
```py
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.title(string.upper())
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```
### Pre-tokenized datasets
- We'll be taking a look at the pre-tokenized version of the IMDB dataset done on **sub words**.
- Unique issues in text classification - sequence of wards can be just as important as their existence.
- **Subwords text encoder** - currently deprecated.
    - It is case sensitive and also retains the punctutations.
    - Use GlobalAveragePooling1D and not Flatten.
    - These subwords are meaningless. They must be put in sequences to so that they have meaningful semantics. 
    - Learning from sequences -> using Recurrent Neural Networks.

---

### Reference Links
- [Embeddings Projector](https://projector.tensorflow.org/)
- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets/catalog/overview)

### Colab
- [IMDB Review classification using Neural Networks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb)
- [Sarcasm Detection using Neural Networks](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb)

### Datasets
- [IMDB Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- [TensorFlow Datasets - Github](https://github.com/tensorflow/datasets/tree/master/docs/catalog)

### Exercise
- [BCC Dataset - Text Classification](BCC_Dataset_Text_Classification.ipynb)

---
