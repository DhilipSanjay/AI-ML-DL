# Natural Language Processing in TensorFlow

**Date:** 18, March, 2021

**Author:** Dhilip Sanjay S

---

[Link to course](https://www.coursera.org/learn/natural-language-processing-tensorflow)

---

## Week 1 - Sentiment in Text
- Load the text, pre process it and set up the data to be fed into the neural network.

### Word based encodings
- Take the character encodings for **each character** in a set. (ASCII values) - But training neural network with these would be a daunting task.
- We'll use **word encodings** - giving numbers for **each word** in a sentence.
- Using Tensorflow Keras API:
    - `tokenizer = Tokenizer(num_words = 100)`
        - It takes the **100 maximum occuring unique words**. But use it carefully, because at times impact of less words can be minimal in training accuracy but huge in training time.
        - It strips the punctutations and removes capitalizations.
    - `tokenizer.fit_on_texts(sentences)`
        - Creates the word encodings.
    - `tokenizer.word_index`
        - Returns key value pair - key is the word and the value is the token for the word.
        - Returns the index of all the words, even above the `num_words` value.

### Text to Sequence
- To feed the text into neural network - Sentences must be of the same length - Just like we would resize the images to same size.
- Using TensorFlow Keras API:
    - `tokenizer.texts_to_sequences(sentences)`
        - If there no entry for a word in the trained encodings, it just skips that word while building the sequence.
        - Here, the `num_words` number of texts will only be used. Other words will be treated as oov.
        - Example: *really* word missing in the training data.
- Thus, we need a large training data to get a broad vocabulary.
- Instead of just ignoring the unseen words, we can put a special value for that unseen word.
- We can do this by passing the `ovv_token = "<OOV>"` argument to the Tokenizer() class initialization.

### Padding
- We need to have some level of uniformity of size - padding is used.
- Using TensorFlow Keras API:
    - `pad_sequences(sequences, padding="post", truncating="post", maxlen=5)`
        - Use it after converting the text to sequences.
        - The list of sentences are formed into a matrix.
        - Post padding -> add padding at the end.
        - Instead of having matrix width to be the length of the longest sentence, you can override it with maxlen. On doing so, you'll lose text in beginning of the sentence.
        - Post truncating - lose text at the end.

### Colab
- [Word encodings - Tokenizer Keras API](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%201.ipynb)
- [Text to seq, Padding - Keras API](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%202.ipynb#scrollTo=rX8mhOLljYeM)
    - Try adding `padding='post', truncating='post'` arguments to the `pad_sequences()` and note the differences.
- [Sarcasm JSON Data Preprocessing](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%201%20-%20Lesson%203.ipynb)

### Datasets
- [News Headlines dataset for sarcasm detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)
- [Common Stop Words](https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js)

### Exercise
- [Explore BBC News Archive](Explore_BBC_News_Archive_Exercise_Answer.ipynb)

---

## Week 2 - Word Embeddings
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

## Week3 - Sequence models
- Relative ordering (sequence of words) matters for the meaning of a sentence.
- For this we'll use specialized neural networks:
    - **RNN** - Recurrent Nerual Networks
        - Context is preserved from time to time. But it might get lost in longer sentences.
    - **GRU** - Gated Recurrent Units
        - Gates are used for controlling the flow of information in the network. 
        - Gates are capable of learning which inputs in the sequence are important and store their information in the memory unit.
    - **LSTM** - Long Short-Term Memory
        - LSTMs have cell states and they carry the context.