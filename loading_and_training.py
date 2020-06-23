#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:58:32 2020

@author: Scott
"""

path = '' #TODO Make this globally accessible

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#from nltk.corpus import stopwords # Need this?
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import nltk
from nltk.translate.bleu_score import sentence_bleu as nltkbleu
#nltk.download('stopwords')
#nltk.download('punkt')

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split


import random
import math
import time

import re
import pickle

import sys

def get_models(load_weights='weights_03_smalltok.hdf5'):

    # =========================================================================
    # Main model
    # =========================================================================

    # Prefix L_ = layer instance
    # Prefix I_ = special input layer instance

    latent_dim_1 = 200
    latent_dim_2 = 200

    ### Encoding layers

    I_enc_in = Input(name='Encoder_input',shape=(max_text_len,))

    L_enc_embed = Embedding(
            name='Encoder_embedding',
            input_dim = vocab_size,
            output_dim = latent_dim_1,
            trainable = True,
            input_length = max_text_len,
            ) #TODO include masking?

    L_enc_lstm = LSTM(
        latent_dim_2,
        return_sequences=False, #TODO true for Attention
        return_state=True,
        name='Encoder_LSTM'
        )

    ### Encoding connections

    enc_embedded = L_enc_embed(I_enc_in)
    enc_out, state_h, state_c = L_enc_lstm(enc_embedded)
    # If return_sequences = True, then enc_out = state_h?

    ### Decoding layers

    #I_dec_in = Input(shape = (max_head_len - 1,))
    I_dec_in = Input(name='dec_ins',shape = (None,))

    L_dec_embed = Embedding(
        vocab_size,
        latent_dim_1,
        input_length = max_head_len - 1 ,
        trainable=True,
        name='Dec_embedding'
        )

    L_dec_lstm = LSTM(
                      latent_dim_2,
                      return_sequences=True,
                      return_state=True,
                      name='Dec_lstm',
                      )

    #TODO TimeDistributed = Necessary? Learn about this!
    L_dec_dense = TimeDistributed(
                                  Dense(vocab_size, activation = 'softmax'),
                                  name='Time_Dist',
                                  )

    ### Decoding connections

    dec_embedded = L_dec_embed(I_dec_in)

    dec_lstm_out, _, _  = L_dec_lstm(
                                    dec_embedded,
                                    initial_state=[state_h, state_c],
                                    )

    dec_out = L_dec_dense(dec_lstm_out)

    ###

    model = Model([I_enc_in, I_dec_in], dec_out)
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # Sparse is not written in tuts, but seems to be required

    # =========================================================================
    # Encoder
    # =========================================================================

    encoder_model = Model(
            inputs = I_enc_in,
            outputs = [enc_out, state_h, state_c]
            )

    # =========================================================================
    # Decoder
    # =========================================================================

    I_decoder_h = Input(shape=(latent_dim_2,))
    I_decoder_c = Input(shape=(latent_dim_2,))
    I_decoder_m = Input(shape=(max_text_len,latent_dim_1))

    dec_embedded_2 = L_dec_embed(I_dec_in)

    dec_lstm_out_2, dec_h_2, dec_c_2  = L_dec_lstm(
                                            dec_embedded_2,
                                            initial_state=[I_decoder_h, I_decoder_c],
                                            )

    dec_out_2 = L_dec_dense(dec_lstm_out_2)

    decoder_model = Model(
            [I_dec_in] + [I_decoder_h, I_decoder_c],
            # With attn: [I_dec_in] + [I_decoder_m, I_decoder_h, I_decoder_c],
            [dec_out_2] + [dec_h_2, dec_c_2]
            )

    # =========================================================================
    # =========================================================================

    # Callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    savepath = path + 'weights_03_smalltok.hdf5'
    checkpoint = ModelCheckpoint(savepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    callbacks = [es,checkpoint]

    # Loading
    if load_weights != None:
        model.load_weights(path+load_weights)

    return model, encoder_model, decoder_model, callbacks


def gen_tokens_maps(return_tokeniser=True):

    splits = get_splits()

    xtr = splits['xtr_df']
    ytr = splits['ytr_df']

    # Without setting a max num_words, there are many words in the vocabulary.
    # This results in massive embedding layers (since they need to map from R2500
    # to something smaller.)
    tokeniser = Tokenizer(
        num_words=2000,
        oov_token = None)#'__unknown__')

    # We want to set it so that the tokenizer does NOT remove _
    index__ = tokeniser.filters.index('_')
    new_filter = tokeniser.filters[:index__] + tokeniser.filters[index__ + 1:]
    tokeniser.filters = new_filter

    tokeniser.fit_on_texts(list(xtr) + list(ytr))

    getword  = tokeniser.index_word
    getindex = tokeniser.word_index

    if return_tokeniser:
        return getword, getindex, tokeniser
    else:
        return getword, getindex

def get_splits(df):
    max_text_len = 300
    max_head_len = 30

    (xtr,
     xcv,
     ytr,
     ycv) = train_test_split(
             df['text'],
             df['headline'],
             test_size=0.05,
             random_state=0,
             shuffle=True)

    _,_,tokeniser = get_tokens_map(return_tokeniser=True)

    ###

    xtr_tok = tokenizer.texts_to_sequences(xtr)
    xcv_tok = tokenizer.texts_to_sequences(xcv)

    ytr_tok = tokenizer.texts_to_sequences(ytr)
    ycv_tok = tokenizer.texts_to_sequences(ycv)

    xtr_tok = pad_sequences(xtr_tok, maxlen=max_text_len, padding='post')
    xcv_tok = pad_sequences(xcv_tok, maxlen=max_text_len, padding='post')

    # Convert list of arrays into 2D array
    xtr_tok = np.stack(xtr_tok)
    xcv_tok = np.stack(xcv_tok)

    ytr_tok = pad_sequences(ytr_tok, maxlen=max_head_len, padding='post')
    ycv_tok = pad_sequences(ycv_tok, maxlen=max_head_len, padding='post')

    #TODO Do we need to stack yxx_tok too?

    vocab_size = len(tokenizer.word_index)+1

    # Here I've changed it. Above is what is given in the tutorial, below is
    # the new one. word_index is hundreds of thousands. It is the pure index
    # of each word. However, these are cut-off at num_words. The +1 is for
    # the special token for unknown words
    vocab_size = tokenizer.num_words + 1

    end_tok = tokenizer.texts_to_sequences(['__end__'])[0][0]

    def set_last_to_end(tok_list, end_tok):
        for i in range(len(tok_list)):
            if tok_list[i][-1] not in [0, end_tok]:
                tok_list[i][-1] = end_tok
        return tok_list

    xtr_tok = set_last_to_end(xtr_tok,end_tok)
    ytr_tok = set_last_to_end(ytr_tok,end_tok)

    def get_model_data(xtok,ytok):
        a = [ xtok, ytok[:,:-1] ] # For enc_in, dec_in
        b = ytok.reshape(ytok.shape[0], ytok.shape[1], 1)[:,1:]
        return a, b

    xtr_tok_modin, ytr_tok_modin = get_model_data(
                xtr_tok,
                ytr_tok
                )

    xcv_tok_modin, ycv_tok_modin = get_model_data(
                xcv_tok,
                ycv_tok
                )

    out = {
            'xtr_df':xtr,
            'ytr_df':ytr,
            'xcv_df':xcv,
            'ycv_df':ycv,

            'xtr_tok':xtr_tok,
            'ytr_tok':ytr_tok,
            'xcv_tok':xcv_tok,
            'ycv_tok':ycv_tok,

            'xtr_tok_modin':xtr_tok_modin,
            'ytr_tok_modin':ytr_tok_modin,
            'xcv_tok_modin':xcv_tok_modin,
            'ycv_tok_modin':ycv_tok_modin,
            }

    return out


def main():
    df = pd.read_pickle(path+'df_split_transformed_June_GCP.pkl')
    start_index = df['cleaned'].sum()
    df = df[:start_index-1] # in case the earler cell was skipped

    model, _, _, callbacks = get_models()

    data_modin = get_splits(df)

    xtr_tok_modin = data_modin['xtr_tok_modin']
    ytr_tok_modin = data_modin['ytr_tok_modin']
    xcv_tok_modin = data_modin['xcv_tok_modin']
    ycv_tok_modin = data_modin['ycv_tok_modin']

    history = model.fit(
                xtr_tok_modin,
                ytr_tok_modin,

                epochs = 12,
                callbacks = [es,checkpoint],
                batch_size = 64,
                validation_data = (xcv_tok_modin, ycv_tok_modin),
                )


if __name__ == '__main__':
    main()










































