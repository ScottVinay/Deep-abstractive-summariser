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
from nltk.tokenize import word_tokenize as word_tokenise

from tensorflow.keras.preprocessing.text import Tokenizer as Tokeniser
from tensorflow.keras.preprocessing.text import one_hot
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

max_text_len = 300
max_head_len = 30
# TODO hyperparam doc?

def get_models(
        load_weights='weights_03_smalltok.hdf5',
        vocab_size=None,
        latent_dim_1=200,
        latent_dim_2=200,
        ):

    # =========================================================================
    # Main model
    # =========================================================================

    # Prefix L_ = layer instance
    # Prefix I_ = special input layer instance

    if vocab_size==None:
        print('Need vocab_size (from get_tokens_map)')
        return

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
        try:
            model.load_weights(path+load_weights)
        except:
            pass

    return model, encoder_model, decoder_model, callbacks


def gen_tokens_maps(df, return_all=True):

    splits = get_df_splits(df)

    xtr = splits['xtr_df']
    ytr = splits['ytr_df']

    # Without setting a max num_words, there are many words in the vocabulary.
    # This results in massive embedding layers (since they need to map from R2500
    # to something smaller.)
    tokeniser = Tokeniser(
        num_words=2000,
        oov_token = None)#'__unknown__')

    # We want to set it so that the tokeniser does NOT remove _
    index__ = tokeniser.filters.index('_')
    new_filter = tokeniser.filters[:index__] + tokeniser.filters[index__ + 1:]
    tokeniser.filters = new_filter

    tokeniser.fit_on_texts(list(xtr) + list(ytr))

    getword  = tokeniser.index_word
    getindex = tokeniser.word_index

    vocab_size = tokeniser.num_words + 1

    if return_all:
        return getword, getindex, vocab_size, tokeniser
    else:
        return getword, getindex, vocab_size

def get_df_splits(df):
    '''
    This is only called from within get_tok_splits and gen_tokens_maps
    '''

    (xtr,
     xcv,
     ytr,
     ycv) = train_test_split(
             df['text'],
             df['headline'],
             test_size=0.05,
             random_state=0,
             shuffle=True)

    return {
            'xtr_df':xtr,
            'xcv_df':xcv,
            'ytr_df':ytr,
            'ycv_df':ycv,
            }

def get_tok_splits(df, tokeniser=None):

    if tokeniser==None:
        _,_,_,tokeniser = get_tokens_map(return_tokeniser=True)

    splits = get_df_splits(df)

    xtr = splits['xtr_df']
    ytr = splits['ytr_df']
    xcv = splits['xcv_df']
    ycv = splits['ycv_df']

    xtr_tok = tokeniser.texts_to_sequences(xtr)
    xcv_tok = tokeniser.texts_to_sequences(xcv)

    ytr_tok = tokeniser.texts_to_sequences(ytr)
    ycv_tok = tokeniser.texts_to_sequences(ycv)

    xtr_tok = pad_sequences(xtr_tok, maxlen=max_text_len, padding='post')
    xcv_tok = pad_sequences(xcv_tok, maxlen=max_text_len, padding='post')

    # Convert list of arrays into 2D array
    xtr_tok = np.stack(xtr_tok)
    xcv_tok = np.stack(xcv_tok)

    ytr_tok = pad_sequences(ytr_tok, maxlen=max_head_len, padding='post')
    ycv_tok = pad_sequences(ycv_tok, maxlen=max_head_len, padding='post')

    #TODO Do we need to stack yxx_tok too?

    # vocab_size = len(tokeniser.word_index)+1    ||| OLD

    # Here I've changed it. Above is what is given in the tutorial, below is
    # the new one. word_index is hundreds of thousands. It is the pure index
    # of each word. However, these are cut-off at num_words. The +1 is for
    # the special token for unknown words

    # vocab_size is calculated AGAIN here, instead of passing in as an
    # argument. This assumes consistency of tokeniser, which SHOULDN'T
    # be an issue, but be aware.
    vocab_size = tokeniser.num_words + 1

    end_tok = tokeniser.texts_to_sequences(['__end__'])[0][0]

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

# Sampling functions

def probSampler(tokens_in):
    tokens = tokens_in[1:] # to account for 0 padding
    chosen = np.random.choice(len(tokens), 1, p=tokens/sum(tokens))[0]
    chosen = chosen + 1 # to account for 0 padding
    return chosen

def suppressedArgmax(tokens, damping=20):
    chosen, _ = beamSuppressedArgmax(tokens, damping=damping, beamwidth=1)
    return chosen[0]

def basicArgmax(tokens):
    chosen, _ = beamSuppressedArgmax(tokens, damping=1, beamwidth=1)
    return chosen[0]

# Beam samplers

def beamSuppressedArgmax(tokens_in, damping=50, beamwidth=3):
    tokens = tokens_in[1:] # account for 0 padding
    tokens[getindex['__end__'] - 1] /= damping

    # The tokens of the words with the greatest conditional probabilities.
    chosenTokens = tokens.argsort()[::-1][:beamwidth]
    # The conditional probabilities themselves.
    chosenProbs = tokens[chosenTokens]
    chosenTokens += 1 # account for 0 padding
    return chosenTokens, chosenProbs

# Translation


def passThroughEncoder(input_text):
    split_in = input_text.split(' ')
    if split_in[0] != '__start__':
        split_in = ['__start__'] + split_in
    if split_in[-1] != '__end__':
        split_in = split_in + ['__end__']

    new_text = ' '.join(split_in)
    input_toks = tokeniser.texts_to_sequences([new_text])
    input_toks = pad_sequences(np.array(input_toks), max_text_len, padding='post')

    e_out, e_h, e_c = encoder_model.predict(input_toks)
    return e_out, e_h, e_c

def neaten(text):
    text = text.replace(' __cm__', ',')
    text = text.replace(' __fs__', '.')
    text = text.replace('__start__ ', '')
    text = text.replace(' __end__', '')
    text = text.strip(' ')
    text = text.strip('\n')
    sents = text.split('. ')
    if len(text)>1:
        text = text[0].upper()+text[1:]
    text = '. '.join(i.capitalize() for i in sents)
    return text

# Greedy

def translate(input_text,
              sampler=suppressedArgmax,
              damping=10):

    e_out, e_h, e_c = passThroughEncoder(input_text)

    decoder_input = np.zeros((1,1))
    decoder_input[0,0] = getindex['__start__']

    output_sequence = []
    out_len = 0
    stopper = False
    while not stopper:

        output_tokens, h, c = decoder_model.predict(
                [decoder_input] + [e_h, e_c]
                )

        # next_token is a number
        output_tokens = output_tokens[0,-1,:]
        next_token = sampler(output_tokens, damping=damping)
        # TODO check if damping is a keyword of sampler

        output_sequence.append(getword[next_token])

        out_len += 1

        if getword[next_token] == '__end__' or out_len >= max_head_len:
            break

        decoder_input[0, 0] = next_token
        e_h, e_c = h, c # why?

    return neaten(' '.join(output_sequence))

# Beam

def beamTranslate(
    input_text,
    width=3,
    finalWidth=3, # this might NEED to be equal to width, have a think
    damping=20,
    sampler=beamSuppressedArgmax,
    lengthBayes=False,
    returnBestOnly=True,
    ):

    e_out, e_h, e_c = passThroughEncoder(input_text)

    decoder_input = np.zeros((1,1))
    decoder_input[0,0] = getindex['__start__']

    currentBeam = [{
        'outseq':['__start__',], # note, this is a list unlike the string in the other func
        'len':0,
        'prob':1,
        'h':e_h,
        'c':e_c,
        }]*1 # start with 1

    finalSentences = []

    stopper = False
    while not stopper:

        newBeam = []

        for cand in currentBeam: # cand = candidate
            decoder_input[0,0] = getindex[cand['outseq'][-1]]
            # h, c change when a word is passed through the decoder,
            # so we need to store these for each possible candidate sentence.
            h = cand['h']
            c = cand['c']

            output_tokens, new_h, new_c = decoder_model.predict(
                    [decoder_input] + [h, c]
                    )
            output_tokens = output_tokens[0,-1,:]

            next3tokens, next3condProbs = sampler(output_tokens, damping=damping)

            for token, conProb in zip(next3tokens, next3condProbs):
                word = getword[token]

                if word=='__end__':
                    finalSentences.append({
                        # ' '.join(...) creates a single sentence from
                        # the list. We apply neaten to make it human-readable.
                        'sent': neaten(' '.join(cand['outseq'])),
                        'prob': cand['prob']*conProb,
                        'len': cand['len']+1,
                    })

                else:
                    new_outseq = cand['outseq'].copy()
                    new_outseq.append(word)

                    newBeam.append({
                        'outseq': new_outseq,
                        'len': cand['len']+1,
                        'prob': cand['prob']*conProb, # = joint prob ya?
                        'h': new_h,
                        'c': new_c
                    })

        if len(finalSentences)>=finalWidth:
            stopper = True
            break

        currentBeam = sorted(newBeam, key=lambda x: x['prob'])[::-1][:width]

    finalSentences = sorted(finalSentences, key = lambda x: x['prob'])[::-1][:width]

    if returnBestOnly:
        return finalSentences[0]['sent']
    else:
        return finalSentences

# Show solutions

def random_translate(printer=0, submode=0, damping=20):
    i=np.random.randint(0,7850) # = len xcv

    intext    = neaten(xcv.loc[xcv.index[i]])
    headline  = neaten(ycv.loc[xcv.index[i]])
    generated = beamTranslate(intext,damping=damping)

    if printer==0:
        return intext, headline, generated

    elif printer==1:
        print(i)
        print('\n')

        print('Text:')
        print(intext)
        print('\n')

        print('Actual headline:')
        print(headline)
        print('\n')

        print('Generated headline:')
        # Include parameter of translate types
        print(generated)
        return

#%%

def bleu(ref,gen,weights=[1,1,1,1]):
    #TODO - remove all punc in both ref and gen
    # Weird way the nltk works:
    # https://stackoverflow.com/questions/40542523/nltk-corpus-level-bleu-vs-sentence-level-bleu-score
    ref = ref.lower()
    gen = gen.lower()
    weights = np.array(weights)
    weights = weights/weights.sum()

    def get_n_toks(toks,n):
        out = []
        splits = toks.split(' ')
        for i in range(0 , len(splits)-n+1):
            list_ngram = splits[i:i+n]
            string_ngram = ' '.join(list_ngram)
            out.append(string_ngram)
        return out

    def bleu_n(reflist, genlist):
        numerator   = 0
        denominator = 0

        print(genlist) ##
        for tok in set(genlist):

            this_num = min(
                reflist.count(tok),
                genlist.count(tok)
                )
            this_den = genlist.count(tok)

            numerator += this_num
            denominator += this_den
        return numerator / denominator

    ans = 0
    for n in range(1,5):
        if min(
                len(ref.split(' ')),
                len(gen.split(' '))
                ) < n:
            break # Avoid looking for n_grams on sentences without n words.

        reftoks = get_n_toks(ref,n)
        gentoks = get_n_toks(gen,n)
        pn = bleu_n(reftoks,gentoks)
        if weights[n-1]!=0:

            if pn==0:
                # This conditional on pn is to avoid log(0) errors.
                pn = 0.0001

            ans += weights[n-1] * np.log(pn)
    ans = np.exp(ans)

    r = len(get_n_toks(ref,1))
    g = len(get_n_toks(gen,1))
    if g <= r:
        mod = np.exp(1 - r/g)
    else:
        mod = 1

    return ans * mod
# Rouge: rouge-L, rouge-1, rouge-n, rouge-s
# F1

#%%



#####

if len(sys.argv) == 1:
    print('Mode required')
    sys.exit()

mode = sys.argv[1]

if mode.lower() in ['lat','0']:
    mode = 0

if mode.lower() in ['translate','1']:
    mode = 1

if mode.lower() in ['bleu_test','2']:
    mode = 2

submode = 0

#####

print('\nLoading dataframe...')
df = pd.read_pickle(path+'df_split_transformed_June_GCP.pkl')
start_index = df['cleaned'].sum()
df = df[:start_index-1]

print('Finding splits...') # Technically duplicated, but not efficient to remove
splits = get_df_splits(df)
xcv = splits['xcv_df']
ycv = splits['ycv_df']

print('Getting tokeniser...')
getword, getindex, vocab_size, tokeniser = gen_tokens_maps(df, return_all=True)

print('Getting tokens of splits...')
data_modin = get_tok_splits(df, tokeniser)

print('Creating model...')
model, encoder_model, decoder_model, callbacks = get_models(vocab_size=vocab_size)

xtr_tok_modin = data_modin['xtr_tok_modin']
ytr_tok_modin = data_modin['ytr_tok_modin']
xcv_tok_modin = data_modin['xcv_tok_modin']
ycv_tok_modin = data_modin['ycv_tok_modin']

if mode==0:
    inp = input('This is fit mode. Are you sure you want to continue? yes/no')
    if inp != 'yes':
        sys.exit()
    print('Beginning fit...')
    history = model.fit(
                xtr_tok_modin,
                ytr_tok_modin,

                epochs = 20,
                callbacks = callbacks,
                batch_size = 64,
                validation_data = (xcv_tok_modin, ycv_tok_modin),
                )
    print('Done')

elif mode==1:
    while True:
        command = input('...')
        if command.lower() in ['quit', 'exit', 'c']:
            break
        random_translate(printer=1, submode=submode)

elif mode==2:

    test_size = 100
    #TODO runtime options

    bleu_avg = 0
    d=20
    for i in range(test_size):
        print(i)
        _, ref, gen = random_translate(printer=0, submode=submode, damping=d)
        bleu_avg += bleu(ref,gen)

    print()
    print(bleu_avg/test_size )





# Note, we are using lots of things as global var names. Easier than passing
# every single thing back and forth











































