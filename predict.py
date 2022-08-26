#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import Adam
import h5py
import sys
import os
from keras import regularizers
from keras.models import load_model

import tensorflow as tf
config = tf.ConfigProto(intra_op_parallelism_threads=16, 
                        inter_op_parallelism_threads=16, 
                        allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

def main():
    
    argc = len(sys.argv)
    argv = sys.argv
    
    nt_onehot=h5py.File(argv[2]+'nt_onehot.mat')
    codon_onehot=h5py.File(argv[2]+'codon_onehot.mat')
    nt_length=np.loadtxt(argv[2]+'nt_length.csv',delimiter=',')
    codon_length=np.loadtxt(argv[2]+'codon_length.csv',delimiter=',')
    nt_onehot=nt_onehot['nt_onehot'][:]
    codon_onehot=codon_onehot['codon_onehot'][:]
   
    a=nt_onehot.shape[0]
    a=np.array(a)
    a=a.astype('int64')
    nt_length=nt_length.astype('int64')
    nt_length=nt_length/1
    codon_length=codon_length.astype('int64')
    codon_length=codon_length/1
    num=a/nt_length

    nt_onehot = nt_onehot.reshape(num, nt_length, 4)
    codon_onehot = codon_onehot.reshape(num, codon_length, 64)

    model=load_model(argv[1])
    predict = model.predict([nt_onehot,codon_onehot])
    np.savetxt(argv[2]+'predict.csv',predict)
    del model
    

if __name__=="__main__":
    main()
