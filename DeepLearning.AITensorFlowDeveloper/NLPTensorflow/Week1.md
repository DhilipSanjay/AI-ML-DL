# Week 1 - Sentiment in Text
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

---

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