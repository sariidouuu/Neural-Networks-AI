# ────── HERE WE COOKIN ──────

import numpy as np

# The following two lines helps the model to understand when a word ends, and when the next one starts
# Without it, python would see an undivided-single line of text and wouldn't be able to process it
import nltk
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer # The algorithm that finds the "root" of a word
stemmer = PorterStemmer() # For stemming


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

# Stemming: a process where a word is converted into its original root form
# When the user sends a prompt, the system stems every word of the sentence
# This helps the model to understand the different ways a word can be written
# We also reduce the volume of the data the model has to demystify (αποστιθήσει)
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

# tokenized_sentence = users prompt !!
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32) # array with zeros
    
    # Checks for each word of the dictionary if it exists in the prompt
    # If yes: put 1 to this position, else 0
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag