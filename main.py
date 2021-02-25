#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' This is a demo file for the PCA model.
    API usage:
        dataloader.load_HDFS(): load HDFS dataset
        feature_extractor.fit_transform(): fit and transform features
        feature_extractor.transform(): feature transform after fitting
        model.fit(): fit the model
        model.predict(): predict anomalies on given data
        model.evaluate(): evaluate model accuracy with labeled data
'''

import sys
from PCA import PCA
from IsolationForest import IsolationForest
from preprocessing import FeatureExtractor
import fasttext
from tokenizer import my_tokenize
import numpy as np
import pandas as pd

ft = fasttext.load_model("fasttext_model/mix.bin")


train_file = "input/training_k3s.txt"
test_file = "input/dec15_k3s.txt"
# train_file = "input/dec15_k3s.txt"
# test_file = "input/training_k3s.txt"
# train_file = "input/newk3s1-train"
# test_file = "input/newk3s1-test"
def get_train_test():
    
    print('Loading phase:')
    train_vecs = load_data(train_file)
    test_vecs = load_data(test_file)
    return train_vecs, test_vecs

def l2_norm(x):
   return np.sqrt(np.sum(x**2))
def div_norm(x):
   norm_value = l2_norm(x)
   if norm_value > 0:
       return x * ( 1.0 / norm_value)
   else:
       return x
def get_sentence_vector(tokens):
    wordvecs_split = [ft[word] for word in tokens]
    processed_word_vectors = []
    for word_vector in wordvecs_split:
        processed_word_vectors.append(div_norm(word_vector))
    processed_word_vectors = np.array(processed_word_vectors)
    return np.mean(processed_word_vectors,axis=0)

def load_data(filepath):
    import random
    print("tokenizing file :" + str(filepath))
    d = []
    with open(filepath, 'r') as fin:
        for idx, line in enumerate(fin):
            d.append(line)
    res = []
    #random.shuffle(d)
    for line in d:
        tokens = my_tokenize(line.lower().rstrip())
        #vecs = [ft[t] for t in tokens if t in ft]
        vecs = get_sentence_vector(tokens)
        res.append(vecs)
    
    return np.array(res)

def save_to_csv(method_name, y_train, y_test):
    if "new" not in train_file:
        train_csv = "input/k3s_train_pred.csv"
        test_csv = "input/k3s_test_pred.csv"
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        df_train[method_name] = y_train
        df_test[method_name] = y_test
        df_train.to_csv(train_csv)
        df_test.to_csv(test_csv)
    else:
        test_csv = "input/merged_predictions.csv"
        df_test = pd.read_csv(test_csv)
        df_test[method_name] = y_test
        df_test.to_csv(test_csv)

def show_res(pred_y):
    res = []
    count = 0
    for idx, y in enumerate(pred_y):
        if y == 1:
            count += 1
            if idx // 10 not in res:
                res.append(idx // 10)
    print("total pred : " + str(count))
    print(res)

def pca():
    ## 1. Load strutured log file and extract feature vectors
    # Save the raw event sequence file by setting save_csv=True
    
    train_vecs,test_vecs = get_train_test()
    print(train_vecs)

    print('Train phase:')
    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(train_vecs, normalization='zero-mean')
    #x_train = train_vecs
    
    ## 2. Train an unsupervised model
    
    # Initialize PCA, or other unsupervised models, LogClustering, InvariantsMiner
    model = PCA() 
    # Model hyper-parameters may be sensitive to log data, here we use the default for demo
    model.fit(x_train)
    # Make predictions and manually check for correctness. Details may need to go into the raw logs
    y_train = model.predict(x_train)
    show_res(y_train)

    ## 3. Use the trained model for online anomaly detection
    print('Test phase:')
    x_test = feature_extractor.transform(test_vecs) 
    #x_test = test_vecs

    # Finally make predictions and alter on anomaly cases
    y_test = model.predict(x_test)
    show_res(y_test)

    # save_to_csv("pca_anomaly", y_train, y_test)

def iforest():
    from sklearn import ensemble
    anomaly_ratio = 0.01
    train_vecs,test_vecs = get_train_test()
    print(train_vecs)

    print('Train phase:')
    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(train_vecs, normalization='zero-mean')
    x_test = feature_extractor.transform(test_vecs)

    #model = ensemble.IsolationForest(contamination=anomaly_ratio)
    model = IsolationForest()
    model.fit(x_train)

    y_train = model.predict(x_train)
    show_res(y_train)

    y_test = model.predict(x_test)
    show_res(y_test)
    save_to_csv("iforest_anomaly", y_train, y_test)

def gmm():
    from sklearn.mixture import GaussianMixture

    train_vecs,test_vecs = get_train_test()
    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(train_vecs)
    x_test = feature_extractor.transform(test_vecs)

    gmm = GaussianMixture(covariance_type="spherical",n_components=600)
    gmm.fit(x_train)
    y_train = gmm.predict(x_train)
    show_res(y_train)

    
    y_test = gmm.predict(x_test)
    show_res(y_test)
    save_to_csv("gmm_anomaly",y_train, y_test)

if __name__ == '__main__':
    
    pca()
    #iforest()
    #gmm()


