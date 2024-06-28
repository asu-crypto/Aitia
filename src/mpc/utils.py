import crypten
import torch
import pandas as pd
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from crypten.encoder import FixedPointEncoder
from kernel_func import RBF_kernel
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

np.random.seed(2022)

def A2B(tensor):
    b = (tensor / tensor.encoder._scale).to(crypten.mpc.binary)
    # b = tensor.to(crypten.mpc.binary)
    b.encoder = FixedPointEncoder(precision_bits=0)
    # print(isinstance(tensor, crypten.mpc.primitives.binary.BinarySharedTensor))
    return b

def B2A(tensor):
    a = tensor.to(crypten.mpc.arithmetic)
    a.encoder = FixedPointEncoder(precision_bits=16)
    return a

def XOR(ts):
    res = crypten.cryptensor(0, ptype=crypten.mpc.binary)
    for v in ts:
        res._tensor = res._tensor ^ v._tensor
    return res

def AND(ts):
    res = crypten.cryptensor(1, ptype=crypten.mpc.binary)
    for v in ts:
        res._tensor = res._tensor & v._tensor
    return res

def predict_batch(x, H, K=RBF_kernel, gamma=0.1):
    # print(x.shape)
    # if isinstance(x, crypten.mpc.MPCTensor):
    #     prediction = torch.zeros(x.shape[0])
    #     prediction = crypten.cryptensor(prediction)
    # else:
    #     prediction = 0
    # vector_a = []
    # vector_b = []
    # for a, b in H.items():
    #     vector_a.append(a)
    #     vector_b.append(b.unsqueeze(dim=0))
    # vector_a = crypten.cat(vector_a)
    # vector_b = crypten.cat(vector_b)
    
    # prediction = (vector_b * K(x, vector_a, gamma)).sum()
    # for a, b in H.items():
    #     if K == RBF_kernel:
    #         prediction = b * K(x[0], a, gamma) + prediction
    #     else:
    #         prediction = b * K(x[0], a) + prediction
    #return prediction
    # alphas = crypten.cat(H.values())
    # vxs = crypten.cat(H.keys())
    # print(vxs.shape,alphas.shape)
    # res = sum([alpha*K(x,vx,gamma) for vx,alpha in H.items()])
    return H.Y.matmul(K(x,H.X,gamma))
    # print(res_matrix.shape,res.shape,res[0].get_plain_text(),res_matrix[0].get_plain_text())
    # return  res #+ self.b

def predict(x, H, K=RBF_kernel, gamma=0.1):
    # alphas = H.Y#crypten.cat(H.Y)
    # vxs = H.X#crypten.cat(H.X)
    # print(vxs.shape,alphas.shape)
    if isinstance(x, crypten.mpc.MPCTensor):
        prediction = torch.zeros(x.shape[0])
        prediction = crypten.cryptensor(prediction)
    else:
        prediction = 0
    vector_a = []
    vector_b = []
    for a, b in H.items():
        vector_a.append(a)
        vector_b.append(b.unsqueeze(dim=0))
    vector_a = crypten.cat(vector_a)
    vector_b = crypten.cat(vector_b)
    prediction = (vector_b * K(x[0], vector_a, gamma)).sum()
    # for a, b in H.items():
    #     if K == RBF_kernel:
    #         prediction = b * K(x[0], a, gamma) + prediction
    #     else:
    #         prediction = b * K(x[0], a) + prediction
    return prediction

def test_mse(H, X, Y, gamma=0.1):
    mse = 0
    # print(Y.shape)
    Y = Y.view(-1)
    # print('start')
    for x, y in tqdm(zip(X, Y)):
        prediction = predict(x, H, gamma=gamma)
        # print(y, prediction)
        mse += (y-prediction) * (y-prediction)
    # print(mse, len(Y))
    # print('end')
    return mse / len(Y)

SECURECI='/home/tson1997/research/bsgdsvr_project'
SVR_DATASET = ['housing', 'cadata', 'cpusmall', 'space_ga']
CSV_DATASET = ['auto-mpg', 'liver_disorder', 'arrhythmia', 'income', 'abalone', 'ncep']

def load_svm_data(name):
    assert (name in SVR_DATASET), f'Unsupported dataset {name}.'
    pardir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    train = load_svmlight_file(SECURECI+'/data/'+name+"_train")
    test = load_svmlight_file(SECURECI+'/data/'+name+"_test")
    train_features = train[0].toarray()
    train_labels = train[1]
    train_data = np.concatenate([train_features, np.reshape(train_labels, [-1, 1])], axis=1)
    test_features = test[0].toarray()
    test_labels = test[1]
    test_data = np.concatenate([test_features, np.reshape(test_labels, [-1, 1])], axis=1)
    return train_data, test_data

def preprocess_auto_mpg():
    df = pd.read_csv(SECURECI+'/data/auto-mpg.data', header=None, delimiter='\s+')
    df[3] = df[3].replace('?', 0).astype(float)
    df = df.iloc[:, [0, 2, 3, 4, 5]].replace('?', 0).astype(float)
    # df = pd.get_dummies(df, prefix='car')
    # for i in [0, 2, 3, 4, 5]:
    #     col_max = df[i].max()
    #     col_min = df[i].min()
    #     df[i] = (df[i]-col_min) / (col_max-col_min)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/auto-mpg_train')
    test.to_csv(SECURECI+'/data/auto-mpg_test')

def preprocess_ncep():
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
    train.to_csv(SECURECI+'/data/ncep_train')
    test.to_csv(SECURECI+'/data/ncep_test')

def preprocess_bupa():
    df = pd.read_csv(SECURECI+'/data/bupa.data', header=None)
    for i in range(6):
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    train = df[df[6]==1].drop(6, axis=1)
    test = df[df[6]==2].drop(6, axis=1)
    train.to_csv(SECURECI+'/data/bupa_train')
    test.to_csv(SECURECI+'/data/bupa_test') 

def preprocess_arrhythmia():
    df = pd.read_csv(SECURECI+'/data/arrhythmia.data', header=None)
    df = df.iloc[:, [0, 2, 3, 14]].replace('?', 0).astype(float)
    for i in [0, 2, 3, 14]:
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    print(df.info)
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/arrhythmia_train')
    test.to_csv(SECURECI+'/data/arrhythmia_test')

def preprocess_census():
    df = pd.read_csv(SECURECI+'/data/census-income.data', header=None)
    df = df.iloc[:, [0, 5, 18]].replace('?', 0).astype(float)
    for i in [0, 5, 18]:
        col_max = df[i].max()
        col_min = df[i].min()
        df[i] = (df[i]-col_min) / (col_max-col_min)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.iloc[:3000]
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(SECURECI+'/data/census-income_train')
    test.to_csv(SECURECI+'/data/census-income_test')
    # print(df.info)

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

def load_csv_data(name):
    train = pd.read_csv(SECURECI+'/data/'+name+'_train')
    test = pd.read_csv(SECURECI+'/data/'+name+'_test')
    train_data= train.iloc[:, 1:].to_numpy()
    test_data = test.iloc[:, 1:].to_numpy()
    return train_data, test_data

def load_data(name):
    if name in SVR_DATASET:
        return load_svm_data(name)
    if name in CSV_DATASET:
        return load_csv_data(name)
    else:
        raise NotImplementedError(f'{name} is not a supported dataset.')

def relative_square_error(pred, label):
     return np.sum(np.square(np.subtract(pred, label))) / np.var(label)

if __name__ == '__main__':
    x = torch.Tensor([0.0, 2.5])
    crypten.init()
    x_enc = crypten.cryptensor(x)
    x_enc_b = A2B(x_enc)
    x_enc_a = B2A(x_enc_b)
    print(x_enc)
    print(x_enc_b)
    print(x_enc_a)
    print(x_enc.get_plain_text())
    print(x_enc_a.get_plain_text())
    print(x_enc_b.get_plain_text())