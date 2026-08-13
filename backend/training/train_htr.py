"""Reference custom CRNN + BiLSTM + CTC architecture for line-level HTR."""
from __future__ import annotations
import argparse, json, os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
HEIGHT=48

def build_model(num_classes:int):
    image=keras.Input(shape=(HEIGHT,None,1),name='image'); x=image
    for filters in (64,128,256):
        x=layers.Conv2D(filters,3,padding='same',activation='relu')(x)
        x=layers.BatchNormalization()(x); x=layers.MaxPool2D((2,2))(x)
    x=layers.Permute((2,1,3))(x)
    x=layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)
    x=layers.Bidirectional(layers.LSTM(256,return_sequences=True,dropout=.2))(x)
    x=layers.Bidirectional(layers.LSTM(256,return_sequences=True,dropout=.2))(x)
    return keras.Model(image,layers.Dense(num_classes+1,name='logits')(x),name='custom_crnn_bilstm_ctc')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--charset',required=True); p.add_argument('--output',default='backend/model/htr_model.keras'); a=p.parse_args()
    with open(a.charset,encoding='utf-8') as f: charset=json.load(f)
    model=build_model(len(charset)); os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True); model.save(a.output)
    print('Architecture saved:',a.output)
    print('For real training, supply padded labels and input/label lengths to tf.nn.ctc_loss.')
if __name__=='__main__': main()
