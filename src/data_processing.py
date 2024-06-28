# BSD 2-Clause License

# Copyright (c) 2021, Lun Wang
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Utility functions. """

import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

SECURECI='/home/tson1997/research/bsgdsvr_project'
CSV_DATASET = ['liver_disorder', 'arrhythmia', 'income', 'abalone', 'ncep']
np.random.seed(2022)

def preprocess_ncep_ncar():
    df1 = pd.read_csv(SECURECI+'/data/pair0043.txt', sep="  ", header=None).replace('?', 0).astype(float)
    df2 = pd.read_csv(SECURECI+'/data/pair0044.txt', sep="  ", header=None).replace('?', 0).astype(float)
    df3 = pd.read_csv(SECURECI+'/data/pair0045.txt', sep="  ", header=None).replace('?', 0).astype(float)
    df4 = pd.read_csv(SECURECI+'/data/pair0046.txt', sep="  ", header=None).replace('?', 0).astype(float)
    df = pd.concat([df1, df2, df3, df4], axis=1)
    df.columns = range(df.columns.size)
    for i in range(8):
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.iloc[:3000]
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/ncep_ncar_train')
    test.to_csv(SECURECI+'/data/ncep_ncar_test')

def preprocess_liver_disorder():
    df = pd.read_csv(SECURECI+'/data/bupa.data', header=None)
    for i in range(6):
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    train, test = train_test_split(df.drop(6, axis=1), test_size=0.2)
    # train = df[df[6]==1].drop(6, axis=1)
    # test = df[df[6]==2].drop(6, axis=1)
    train.to_csv(SECURECI+'/data/liver_disorder_train')
    test.to_csv(SECURECI+'/data/liver_disorder_test') 

def preprocess_arrhythmia():
    df = pd.read_csv(SECURECI+'/data/arrhythmia.data', header=None)
    df = df.iloc[:, [0, 2, 3, 14]].replace('?', 0).astype(float)
    for i in [0, 2, 3, 14]:
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/arrhythmia_train')
    test.to_csv(SECURECI+'/data/arrhythmia_test')

def preprocess_income():
    df = pd.read_csv(SECURECI+'/data/census-income.data', header=None)
    df = df.iloc[:, [0, 5, 18]].replace('?', 0).astype(float)
    for i in [0, 5, 18]:
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.iloc[:3000]
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/income_train')
    test.to_csv(SECURECI+'/data/income_test')

def preprocess_abalone():
    df = pd.read_csv(SECURECI+'/data/abalone.data', header=None)
    df = df.iloc[:, [1,2,3,4,5,6,7,8]].replace('?', 0).astype(float)
    for i in [1,2,3,4,5,6,7,8]:
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    df = df.iloc[np.random.permutation(len(df))]
    # df = df.iloc[:3000]
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/abalone_train')
    test.to_csv(SECURECI+'/data/abalone_test')

def load_data(name):
    assert (name in CSV_DATASET), f'Dataset {name} not supported.'
    train = pd.read_csv(SECURECI+'/data/'+name+'_train')
    test = pd.read_csv(SECURECI+'/data/'+name+'_test')
    train_data= train.iloc[:, 1:].to_numpy()
    test_data = test.iloc[:, 1:].to_numpy()
    return train_data, test_data

def check_ci_pair(name, feature, target):
    valid_data = ['abalone', 'arrhythmia']

def relative_square_error(pred, label):
     return np.sum(np.square(np.subtract(pred, label))) / np.var(label)

if __name__ == '__main__':
    # preprocess_auto_mpg()
    # preprocess_ncep()
    # preprocess_bupa()
    # train_data, test_data = load_data('bupa')
    # print(train_data.shape, test_data.shape)
    # print(test_data[:, 0].max(), test_data[:, 0].min(), train_data[:, 5].min())
    # print(train_features, train_labels, test_features, test_labels)

    # print(relative_square_error([1, 2], [1.1, 1.9]))
    # preprocess_auto_mpg()
    preprocess_abalone()
    preprocess_ncep_ncar()
    preprocess_liver_disorder()
    preprocess_arrhythmia()
    preprocess_income()
