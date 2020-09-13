## Move all this into loading and training




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 23:02:48 2020

@author: Scott
"""


from loading_and_training import get_models

df =

model, encoder_model, decoder_model, callbacks = get_models()
getword, getindex, vocab_size, tokeniser = gen_tokens_maps(df, return_all=True)

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
    input_toks = tokenizer.texts_to_sequences([new_text])
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

def translate(input_text, sampler=suppressedArgmax, damping=10):
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

def show_new():
    i=np.random.randint(0,7850) # = len xcv

    print(i)
    print('\n')


    print('Text:')
    print(neaten(xcv.loc[xcv.index[i]]))
    print('\n')

    print('Actual headline:')
    print(neaten(ycv.loc[xcv.index[i]]))
    print('\n')

    print('Generated headline:')
    # Parameter of translate types
    print( translate(xcv.loc[xcv.index[i]], damping=1) )

    return