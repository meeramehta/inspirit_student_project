import pandas as pd
import streamlit as st
import numpy as np
import nltk
import matplotlib.pyplot as plt
import re
import spacy
# import tensorflow as tf
# import tensorflow_datasets as tfds
# from tensorflow.keras import layers , activations , models , preprocessing, utils
# from keras import Input, Model
# from keras.activations import softmax
# from keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.optimizers import RMSprop
# from keras_preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from keras_preprocessing.text import Tokenizer
# tf.random.set_seed(1)
import time
import os
import gdown

# set pandas viewing options
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option("max_colwidth", None)
#pd.reset_option("max_colwidth")

# the source of our data is: https://github.com/nbertagnolli/counsel-chat

# load our weights
# !wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Mental%20Health%20Chatbots/chatbot_seq2seq_v3.h5"

st.set_page_config( page_title="Inspirit AI Project: Mental Health Chatbots", )
st.title("Chatbot 1: Rule Based")
hello_keywords = ['hello', 'goodmorning', 'hi', 'hey', 'whats up', 'goodevening', 'morning', 'evening']
goodbye_keywords = ['bye', 'stop', 'goodbye', 'quit', 'thank']
hello_response = 'Hi, nice to meet you, how are you feeling today?'
goodbye_response = 'I hope I was able to help, comeback anytime.'
default_response = 'That is intresting, could you give me more information'
Pfeelings_keyword = ['excited', 'happy', 'proud', 'motiavted', 'thrilled' ]
Nfeelings_keyword =['sad', 'deepresed', 'unhappy', 'angry', 'frustrated', 'disappointed']
not_feelings = ['not']
pos_response ="Great to hear that you are feeling "
neg_response = "Why do you think that you are feeling "
repeat_response = "Why are you not "
not_response = "Why are you "
secoundN_feelings = ['dead', 'failing', 'lost', 'trouble', 'loss', 'insecurity', 'bullied', 'health', 'cancer', 'old', 'died', 'passed' ]
secoundN_response = "I am sorry to hear that, can you tell me more? "
secoundP_feeling = ['good', 'well', 'job', 'healthy', 'dream', 'graduated', 'rasie', 'promoted', 'grade', 'birthday']
feeling_keywords = ["feel", "felt", "feeling"]
thirdN_feeling = ["no", "dont't"]
thirdP_feeling =['yes']
thirdP_response =['yes']
thirdPN_response =['no']
secoundP_response = "Congratulations, I am glad to here that!! "
whatelse = "Are you going to celebrate?"
yes = "Have fun!"
no = "That's fine, I hope you have a great rest of your day!! "
last = "I am sorry to hear that this has happend to you, at this moment I have not been programed to handle this if it is serious I suggest reaching out to a trained theropist. "
Pfeelings_keyword = ['excited', 'happy', 'proud', 'motiavted', 'thrilled', 'well' ]
Nfeelings_keyword =['sad', 'deepresed', 'unhappy', 'angry', 'frustrated', 'disappointed']
flaged_words = ['suicied', 'killing', 'ending', 'kill', 'slit', 'jump', 'hang']
not_feelings = ['not']
pos_response ="Great to hear that you are feeling "
neg_response = "Why do you think that you are feeling "
repeat_response = "Why are you not "
not_response = "Why are you "
secoundN_feelings = ['dead', 'failing', 'lost', 'trouble', 'loss', 'insecurity', 'bullied', 'health', 'cancer', 'old', 'died', 'passed' ]
secoundN_response = "I am sorry to hear that, can you tell me more? "
secoundP_feeling = ['good', 'job', 'healthy', 'dream', 'graduated', 'rasie', 'promoted', 'grade', 'birthday']
feeling_keywords = ["feel", "felt", "feeling"]
thirdN_feeling = ["no", "dont't"]
thirdP_feeling =['yes']
thirdP_response =['yes']
thirdPN_response =['no']
youneed = "If you need help or anyone to helo you call: 988 or go to this site to chat, https://988lifeline.org/ "
secoundP_response = "Congratulations, I am glad to here that!! "
whatelse = "Are you going to celebrate?"
yes = "Have fun!"
no = "That's fine, I hope you have a great rest of your day!! "
last = "Tell me more. I am here to help but I suggest reaching out to a trained therapist for more serious issues. "

# Emotion Keywords and Responses

# Anxiety, nervousness 
anx_response = "you sound anxious, I can recommend a few breathing exercises to calm down linked here: https://www.medicalnewstoday.com/articles/breathing-exercises-for-anxiety."
anx_keywords = ["anxious" , "nervous" , "worried" , "worry" , "worrisome" , "anxiety", 'nervousness' , "scared"]

# Bullying
bully_keywords = ["bully", "bullied","bullying", "abused", "abusing", "hurt", "agressive", "violent", "hit", "punched", "degrade"]
bully_response = "If you are facing forms of bullying or abuse, please reach out to the national domestic violence hotline here: https://www.thehotline.org/"

# Panic Attacks
panic_response = "It sounds like you might be experiencing a high level of stress. Why you are feeling this way? Please refer to this article for guidance: https://www.medicalnewstoday.com/articles/321510#methods "
panic_keywords = ["stressed", "panic", "stress", "fear"]

def tokenize(input):
  tokenizer = nltk.RegexpTokenizer(r"\w+")
  tokens = tokenizer.tokenize(input)## YOUR CODE HERE
  return tokens
# Rule-Based ChatBot
def chatbot_3(input):
  ## BEGIN YOUR CODE HERE
  input = input.lower()
  # Tokenize input and remove punctuation
  input_tokens = tokenize(input)
  
  if flaged_words in input_tokens:
    return youneed

  if 'no' in input_tokens and len(input_tokens) == 1:
    return no 

    # Respond to hello keyword if present
  for keywords in  hello_keywords:
   if keywords in input_tokens:
    return hello_response

  for keyword in bully_keywords:
    if keyword in input_tokens:
      return bully_response
  for keyword in panic_keywords:
    if keyword in input_tokens:
      return panic_response
  for keyword in anx_keywords:
    if keyword in input_tokens:
      return anx_response

  for keyword in Nfeelings_keyword:
        if keyword in input_tokens:
          return neg_response + keyword + "?"   
        if input_tokens[0:2] == ["i", "am"]:
  # Iterate through the input_tokens list and find the first emotion keyword
            return not_response + "  ".join(input_tokens[2:]) + "?"
     
      

  for keyword in Pfeelings_keyword:
    if keyword in input_tokens:
      return pos_response + keyword + "! Why are you feeling " + keyword + "?"

  for keyword in secoundP_feeling:
    if keyword in input_tokens:
      return secoundP_response + whatelse
  for keyword in thirdP_feeling:
      if keywords in input_tokens:
          return yes
       
    

  for keywords in  goodbye_keywords:
    if keywords in input_tokens:
      return goodbye_response # Respond to goodbye keyword if present
  for keywords in secoundN_feelings:
    if keywords in input_tokens:
      return secoundN_response
  for keywords in thirdN_feeling:
    if keywords in input_tokens:
        return no
    else:
        return last

name= "RuleBased_ChatBot"     
prompt=st.text_input("Write to the chatbot. Press ENTER to see chatbot's response")

response_str=chatbot_3(prompt)
st.write(f"{name}'s response: {response_str}")
st.caption("This chatbot will also provide resources if it detects depression, stress, or abuse.")
st.code(
"""# Rule-Based ChatBot
def chatbot_3(input):
  input = input.lower()
  # Tokenize input and remove punctuation
  input_tokens = tokenize(input)
  
  if flaged_words in input_tokens:
    return youneed

  if 'no' in input_tokens and len(input_tokens) == 1:
    return no 

    # Respond to hello keyword if present
  for keywords in  hello_keywords:
   if keywords in input_tokens:
    return hello_response



  for keyword in Nfeelings_keyword:
        if keyword in input_tokens:
          return neg_response + keyword + "?"   
        if input_tokens[0:2] == ["i", "am"] and input_tokens[2] == "not":
  # Iterate through the input_tokens list and find the first emotion keyword
            for i, token in enumerate(input_tokens[3:]):
              if token.lower() in Pfeelings_keyword:
                feeling = input_tokens[i+3:]
              return repeat_response + "  ".join(feeling) + "?"
     
      

  for keyword in Pfeelings_keyword:
    if keyword in input_tokens:
      return pos_response + keyword + "! Why are you feeling " + keyword + "?"

  for keyword in secoundP_feeling:
    if keyword in input_tokens:
      return secoundP_response + whatelse
  for keyword in thirdP_feeling:
      if keywords in input_tokens:
          return yes
       
    

  for keywords in  goodbye_keywords:
    if keywords in input_tokens:
      return goodbye_response # Respond to goodbye keyword if present

  for keywords in secoundN_feelings:
    if keywords in input_tokens:
      return secoundN_response
  for keywords in thirdN_feeling:
    if keywords in input_tokens:
        return no
    else:
        return last"""
)