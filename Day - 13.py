# -*- coding: utf-8 -*-
"""Day - 13.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1N8mMCOhmGCeRJgzANfSUVpNpSHr32QgI
"""

import spacy

def pos_tagging(sentence):
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the sentence
    doc = nlp(sentence)

    # Extract and print each token with its part-of-speech tag
    for token in doc:
        print(f"{token.text}: {token.pos_}")

# Test the function with the given sentence
test_sentence = 'NLP is amazing and fun to learn.'
pos_tagging(test_sentence)