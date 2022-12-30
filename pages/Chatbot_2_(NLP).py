import pandas as pd
import streamlit as st
import numpy as np
import nltk
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers , activations , models , preprocessing, utils
from keras import Input, Model
from keras.activations import softmax
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
tf.random.set_seed(1)
import time
import os


# set pandas viewing options
pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)
pd.set_option("max_colwidth", None)
#pd.reset_option("max_colwidth")

# the source of our data is: https://github.com/nbertagnolli/counsel-chat

import subprocess

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

st.set_page_config( page_title="Inspirit AI Project: Mental Health Chatbots", )
st.title("Chatbot 2: NLP Chatbot (LSTM)")

# load our weights
runcmd("wget -q https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Mental%20Health%20Chatbots/chatbot_seq2seq_v3.h5")
chat_data = pd.read_csv("https://raw.githubusercontent.com/nbertagnolli/counsel-chat/master/data/20200325_counsel_chat.csv")

# Model Code

X = chat_data["questionText"]
y = chat_data["answerText"]

def preprocess_text(phrase): 
  phrase = re.sub(r"\xa0", "", phrase) # removes "\xa0"
  phrase = re.sub(r"\n", "", phrase) # removes "\n"
  phrase = re.sub("[.]{1,}", ".", phrase) # removes duplicate "."s
  phrase = re.sub("[ ]{1,}", " ", phrase) # removes duplicate spaces

  return phrase

X = X.apply(preprocess_text)
y = y.apply(preprocess_text)

question_lengths, answer_lengths = [], []

for (question, answer) in zip(X, y): 
  # split by "."
  question_arr = question.split(".")
  answer_arr = answer.split(".")

  # get length
  length_question = len(question_arr)
  length_answer = len(answer_arr)

  # add to array
  question_lengths.append(length_question)
  answer_lengths.append(length_answer)

# NLP code
question_answer_pairs = []

MAX_LENGTH = 100 # the maximum length for our sequences

for (question, answer) in zip(X, y):
  question = preprocess_text(question) 
  answer = preprocess_text(answer)

  # split up question and answer into their constituent sentences

  question_arr = question.split(".")
  answer_arr = answer.split(".")

  # get the maximum number of question/answer pairs we can form,
  # which will be the shorter of len(question_arr) and len(answer_arr)

  max_sentences = min(len(question_arr), len(answer_arr))

  for i in range(max_sentences):
    q_a_pair = []

    # get maximum sentence length
    max_q_length = min(MAX_LENGTH, len(question_arr[i]))
    max_a_length = min(MAX_LENGTH, len(answer_arr[i]))

    # append question, answer to pair (e.g,. first sentence of question + first sentence of answer, etc.)
    question_to_append = question_arr[i][0:max_q_length]
    q_a_pair.append(question_to_append)

    answer_to_append = "<START> " + answer_arr[i][0:max_a_length] + " <END>"
    q_a_pair.append(answer_to_append)

    question_answer_pairs.append(q_a_pair)

def tokenize(sentence):
  tokens = sentence.split(" ")
  return tokens

def tokenize_and_pad(sentence, max_len):
  """
    Tokenizes our sentence (splits up the individual words), adds <SOS> and <EOS> tags, 
    and, if it's too short, add, padding at the end (before <EOS>)
  """

  sentence_arr = sentence.split(" ")

  diff = max_len - (len(sentence_arr) + 2)

  if diff > 0: # if too short, add padding + start/end tokens
    tokenized_sentence = ["<SOS>"] + sentence_arr + ["<EOS>"] + (["<pad>"] * diff) 
  elif diff == 0: # if right length, just add start/end tokens
    tokenized_sentence = ["<SOS>"] + sentence_arr + ["<EOS>"]
  else: # if too long, add start/end tokens, truncate
    tokenized_sentence = ["<SOS>"] + sentence_arr[0:diff] + ["<EOS>"]

  return tokenized_sentence
tf.keras.backend.clear_session()
  # re-create questions, answers
questions = [arr[0] for arr in question_answer_pairs]
answers = [arr[1] for arr in question_answer_pairs]

target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
tokenizer = Tokenizer(filters=target_regex)
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# create encoder input data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])

# create decoder input data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])


enc_inputs = Input(shape=(None,))
enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inputs)
enc_lstm = LSTM(200, return_state=True)
_, state_h, state_c = enc_lstm(enc_embedding)
enc_states = [state_h, state_c]

dec_inputs = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inputs)
dec_lstm = LSTM(200, return_state=True, return_sequences=True)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

dec_dense = Dense(VOCAB_SIZE, activation=softmax)
output = dec_dense(dec_outputs)


model = Model([enc_inputs, dec_inputs], output)

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

# model.summary()

tf.keras.backend.clear_session()

path_to_weight = "chatbot_seq2seq_v3.h5"

model.load_weights(path_to_weight)

def make_inference_models():
    dec_state_input_h = Input(shape=(200,), name="anotherlayer1")
    dec_state_input_c = Input(shape=(200,), name="anotherlayer2")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                             initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
        inputs=[dec_inputs] + dec_states_inputs,
        outputs=[dec_outputs] + dec_states)
    #print('Inference decoder:')
    #dec_model.summary()
    #print('Inference encoder:')
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
    #enc_model.summary()
    return enc_model, dec_model

def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != '':
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                         maxlen=maxlen_questions,
                         padding='post')
    

enc_model, dec_model = make_inference_models()

question = st.text_input("Write to the AI powered chatbot. Press ENTER to see chatbot's response. (Response will take a few seconds to load)")

for _ in range(1):
    states_values = enc_model.predict(
        str_to_tokens(question))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq]
                                              + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' \
                or len(decoded_translation.split()) \
                > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

st.write("LSTM Chatbot response: ", decoded_translation)

st.subheader("Here is a look at some of the training data we used from Counsel Chat:")
st.write("Data source: https://github.com/nbertagnolli/counsel-chat")
st.write(chat_data.head(20)[['questionText', 'topic', 'answerText']])
st.write("The questionText column contains examples of patient's questions to their therapist.")
st.write("The answerText column contains examples of therapists responses.")

st.subheader("Plotting popular topics from our training data set")

topics = chat_data["topic"].unique()
lengths = chat_data["topic"].value_counts()
fig = plt.figure(figsize = (10, 5))
plt.bar(x = lengths.keys(), height = lengths.values)
ax = plt.gca()
for tick in ax.get_xticklabels():
    tick.set_rotation(80)

st.pyplot(fig)
st.caption("""The training data contains a lot more questions about depression, anxiety and counseling fundamentals, 
than questions about military issues and human-sexuality. We expect our model to do better on 
the topics for which it has more data.""")
# Display the plot
st.subheader("Lengths of questions (in red) and answers (in blue)")
fig = plt.figure(figsize = (10, 5))
plt.hist(question_lengths, color = "red", alpha = 0.5)
plt.hist(answer_lengths, color = "blue", alpha = 0.2)
plt.axvline(np.mean(question_lengths), color = "red")
plt.axvline(np.mean(answer_lengths), color = "blue")
st.pyplot(fig)
st.caption("""The training data contains longer answer text than question text. The red line is the average length in number of 
sentences of the question, and the blue line is the average length in number of sentences of the answers.""")

st.subheader("LSTM Model Architecture")
st.code(
    """enc_inputs = Input(shape=(None,))
  enc_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(enc_inputs)
  enc_lstm = LSTM(200, return_state=True)
  _, state_h, state_c = enc_lstm(enc_embedding)
  enc_states = [state_h, state_c]

  dec_inputs = Input(shape=(None,))
  dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)(dec_inputs)
  dec_lstm = LSTM(200, return_state=True, return_sequences=True)
  dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)

  dec_dense = Dense(VOCAB_SIZE, activation=softmax)
  output = dec_dense(dec_outputs)


  model = Model([enc_inputs, dec_inputs], output)

  model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

  def make_inference_models():
    dec_state_input_h = Input(shape=(200,), name="anotherlayer1")
    dec_state_input_c = Input(shape=(200,), name="anotherlayer2")
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                            initial_state=dec_states_inputs)
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
      inputs=[dec_inputs] + dec_states_inputs,
      outputs=[dec_outputs] + dec_states)
    print('Inference decoder:')
    dec_model.summary()
    print('Inference encoder:')
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
    enc_model.summary()
    return enc_model, dec_model

  enc_model, dec_model = make_inference_models()
  """
)