#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Based on: 
https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras/blob/master/train_bot.py
'''
from __future__ import print_function

import argparse
import json
import random
import re
import os
import sys
from collections import Counter

from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.layers.merge import concatenate
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
import keras.backend as K
import theano.tensor as T
import numpy as np
import cPickle as pickle

BOS=2
EOS=3
UNK=4

np.random.seed(42)  # for reproducibility
r_word=re.compile('(\W+)?')


# Convierte palabras a indices
def tokenize(sent):
    return [x.strip().lower() for x in r_word.split(sent) if x.strip()]

# Dada una pregunta genera una respuesta, una palabra a la vez
def print_result(input,model,i2wi,maxlen_input):
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, 0] = BOS  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        mp = np.argmax(ye)
        #print(mp,ans_partial)
        #ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, k+1] = mp
    text = []
    for k in ans_partial[0]:
        k = k.astype(int)
        w = i2w[k]
        text.append(w)
    return(" ".join(text))

# Función principal (interfaz con línea de comandos)
if __name__ == '__main__':
    p = argparse.ArgumentParser("conversations")
    p.add_argument("JSON",
            action="store",
            help="Json file with conversations")
    p.add_argument("--mode",default="train",
            action="store", dest="mode",
            help="Mode of execution: train, test, clean [train]")
    p.add_argument("--data",default="data.pckl",type=str,
            action="store", dest="data",
            help="Data model [data.pckl]")
    p.add_argument("--model",default="model.h5",type=str,
            action="store", dest="model",
            help="Weights model [model.h5]")
    p.add_argument("--voca",default="voca.pckl",type=str,
            action="store", dest="voca",
            help="Voca model [voca.pckl]")
    p.add_argument("--epochs",default=50,type=int,
            action="store", dest="epochs",
            help="Epochs [10]")
    p.add_argument("--maxlen_input",default=10,type=int,
            action="store", dest="maxlen_input",
            help="Maximum lenght input [20]")
    p.add_argument("--word_embedding_size",default=20,type=int,
            action="store", dest="word_embedding_size",
            help="Word embedding size [30]")
    p.add_argument("--sentence_embedding_size",default=40,type=int,
            action="store", dest="sentence_embedding_size",
            help="Sentence embedding size [60]")
    p.add_argument("--min_count",default=0,type=int,
            action="store", dest="min_count",
            help="Min count in words [0]")
 
    p.add_argument("--train_size",default=0.9,type=float,
            action="store", dest="train_size",
            help="Size of training between 0.0 an 1.0 [0.9]")
    p.add_argument("-v", "--verbose",
            action="store_true", dest="verbose",
            help="Verbose mode [Off]")

    opts = p.parse_args()

    # Si el modo es clean, se borra modelo, vocabulario y datos
    if opts.mode.startswith('clean'):
        os.remove(opts.model)
        os.remove(opts.voca)
        os.remove(opts.data)
        sys.exit()
     

    # Abre conversaciones
    with open(opts.JSON) as outfile:
        conversations=json.load(outfile)


    # Separa de forma aleatoria las conversaciones
    ids=[x for x in range(len(conversations))]
    random.shuffle(ids)
    train=[ conversations[x] for x in ids[:int(opts.train_size*len(ids))]]
    test =[ conversations[x] for x in ids[int(opts.train_size*len(ids)):]]

    print("Random train",len(train))
    print("Reandom test",len(test))

    # Crea vocabulario para representar a las palabras con indices
    if os.path.isfile(opts.voca):
        voca = pickle.load( open(opts.voca, "rb" ))
        i2w=voca['i2w']
        w2i=voca['w2i']
    else:
        voca=Counter()
        for conv in train:
            for msg,ans in conv:
                msg=tokenize(msg)
                ans=tokenize(ans)
                voca.update(msg)
                voca.update(ans)

        i2w={BOS:'BOS',EOS:'EOS',UNK:'unk',0:' '}
        w2i={'BOS':BOS,'EOS':EOS,'unk':UNK}
        i=UNK+1
        for w,c in voca.most_common():
            if c>opts.min_count:
                i2w[i]=w
                w2i[w]=i
                i+=1
        pickle.dump({'i2w':i2w,'w2i':w2i}, open(opts.voca, "wb" ) )
    print("Vocabulary size", len(i2w))

    # Si ya existe un archivo con los datos recuperar
    if os.path.isfile(opts.data):
        data = pickle.load( open(opts.data, "rb" ))
        train_msg=data['train_msg']
        train_ans=data['train_ans']
        test_msg=data['test_msg']
        test_ans=data['test_ans']

    # Caso contrario convertir oraciones en matrices
    else:
        train_msg,train_ans=[],[]
        for i, item in enumerate(train):
            for msg,ans in item:
                msg = [BOS]+[w2i[w] if w in w2i else UNK for w in tokenize(msg)]+[EOS]
                ans = [BOS]+[w2i[w] if w in w2i else UNK for w in tokenize(ans)]+[EOS]
                train_msg.append(np.asarray(msg))
                train_ans.append(np.asarray(ans))
           
        test_msg,test_ans=[],[]
        for i, item in enumerate(test):
            for msg,ans in item:
                msg = [BOS]+[w2i[w] if w in w2i else UNK for w in tokenize(msg)]+[EOS]
                ans = [BOS]+[w2i[w] if w in w2i else UNK for w in tokenize(ans)]+[EOS]
                test_msg.append(np.asarray(msg))
                test_ans.append(np.asarray(ans))

        train_msg = sequence.pad_sequences(train_msg, maxlen=opts.maxlen_input)
        train_ans = sequence.pad_sequences(train_ans, maxlen=opts.maxlen_input,padding="post")
        test_msg = sequence.pad_sequences(test_msg, maxlen=opts.maxlen_input)
        test_msg = sequence.pad_sequences(test_ans, maxlen=opts.maxlen_input,padding="post")
        pickle.dump({'train_msg':train_msg,'train_ans':train_ans,'test_msg':test_msg,'test_ans':test_ans}, open(opts.data, "wb" ) )

   
    # Crear vectores de embeddings
    dictionary_size=len(i2w)+UNK+1
    embedding_matrix = np.zeros((dictionary_size, opts.word_embedding_size))

    ad = Adam(lr=0.00005) 
    # Configuración de modelo
    # Entrada mensaje usuario
    input_context    = Input(shape=(opts.maxlen_input,), dtype='int32', name='input_context')
    # Respuesta chatbot 
    input_answer     = Input(shape=(opts.maxlen_input,), dtype='int32', name='input_answer')
    # Capa de embeddings
    Shared_Embedding = Embedding(output_dim=opts.word_embedding_size,input_dim=dictionary_size, weights=[embedding_matrix], input_length=opts.maxlen_input)
    # LSTM codificadora 
    LSTM_encoder     = LSTM(opts.sentence_embedding_size, kernel_initializer= 'lecun_uniform')
    # LSTM decodificadora 
    LSTM_decoder     = LSTM(opts.sentence_embedding_size, kernel_initializer= 'lecun_uniform')
    
    # La entrada se conecta a los embeddings (de indices a embeddings) 
    word_embedding_context = Shared_Embedding(input_context)
    # Se pasa por la primera LSTM 
    context_embedding = LSTM_encoder(word_embedding_context)

    # La entrada se conecta a los embeddings (de indices a embeddings) 
    word_embedding_answer = Shared_Embedding(input_answer)
    # Se pasa por la segunda LSTM 
    answer_embedding = LSTM_decoder(word_embedding_answer)

    # Se mezcla el resumen de la codificadora, con el resumen de la
    # decodificadora
    merge_layer = concatenate([context_embedding, answer_embedding], axis=1)
    # Se  conecta con una capa que representa a la palabra que se prende
    out = Dense(dictionary_size/2, activation="relu")(merge_layer)
    out = Dense(dictionary_size, activation="softmax")(out)
   
    # Se crea modelo, se compila, y si hay parámetros (pesos) se recuperan
    model = Model(input=[input_context, input_answer], output = [out])
    model.compile(loss='categorical_crossentropy', optimizer=ad)

    if os.path.isfile(opts.model):
        model.load_weights(opts.model)

    print(model.summary())



    print('Number of exemples = %d'%(len(train)))
    num_subsets=1
    step = np.around(len(train_msg)/num_subsets)
    round_exem = step * num_subsets

    x = range(0,opts.epochs) 
    valid_loss = np.zeros(opts.epochs)
    train_loss = np.zeros(opts.epochs)


    import sys
    # Prueba iterativa 
    if opts.mode.startswith('test'):
        while True:
            msg = raw_input("> ")
            if msg.startswith("exit"):
                sys.exit()
            msg = [BOS]+[w2i[w] if w in w2i else UNK for w in tokenize(msg)]+[EOS]
            msg = np.asarray([msg])
            msg= sequence.pad_sequences(msg, maxlen=opts.maxlen_input)
            ans=print_result(msg,model,i2w,opts.maxlen_input)
            ans=ans.replace("BOS","")
            ans=ans.replace("EOS","")
            print(ans.strip())

        
    # Entrenamiento
    for m in range(opts.epochs):
	for n in range(0,round_exem,step):
	    
	    q2 = train_msg[n:n+step]
	    s = q2.shape
	    count = 0
	    for i, sent in enumerate(train_ans[n:n+step]):
		l = np.where(sent==EOS)
		limit = l[0][0]
		count += limit + 1
	    Q = np.zeros((count,opts.maxlen_input))
	    A = np.zeros((count,opts.maxlen_input))
	    Y = np.zeros((count,dictionary_size))
	    
	    # Loop over the training examples:
	    count = 0
	    for i, sent in enumerate(train_ans[n:n+step]):
		ans_partial = np.zeros((1,opts.maxlen_input))
		
		# Loop over the positions of the current target output (the current output sequence):
		l = np.where(sent==EOS)  #  the position of the symbol EOS
		limit = l[0][0]

		for k in range(1,limit+1):
		    # Mapping the target output (the next output word) for one-hot codding:
		    y = np.zeros((1, dictionary_size))
		    y[0, sent[k]] = 1

		    # preparing the partial answer to input:
		    ans_partial[0,:k] = sent[0:k]
		    # training the model for one epoch using teacher forcing:
		    
		    Q[count, :] = q2[i:i+1] 
		    A[count, :] = ans_partial 
		    Y[count, :] = y
                    #print(q2[i:i+1],ans_partial)
		    count += 1
		    
	    print('Training epoch: %d, training examples: %d - %d'%(m,n, n + step))
	    model.fit([Q, A], Y, batch_size=100, epochs=1)
        
            res=print_result(train_msg[0:1],model,i2w,opts.maxlen_input)
            print(res)
            res=print_result(test_msg[0:1],model,i2w,opts.maxlen_input)
            print(res)

    model.save_weights(opts.model, overwrite=True)
