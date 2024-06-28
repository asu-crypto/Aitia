import numpy as np
import random
import crypten
from tqdm import tqdm
from crypten.config import cfg
import torch
from tqdm import tqdm
import time
class BSGD():
    def __init__(self, budget_size, max_iters, dataset_size, learning_rate=0.05,margin=1e-3, kernel_func = "rbf",mode="normal",verbose=False):
        self.B = budget_size
        self.max_iters = max_iters
        self.mode = mode
        self.lr = learning_rate
        self.eps = margin
        size = int(dataset_size*budget_size)
        feature_size = 1
        self.beta  = 0
        self.verbose=verbose
        if self.mode == "normal":
            self.vectors = torch.zeros((size+1,feature_size),dtype=torch.float64)
            self.weights = torch.zeros(size+1,dtype=torch.float64)
            self.IDX = torch.arange(1,size+2,dtype=torch.float64)*0
            
            self.buffer = torch.zeros(size+1,dtype=torch.float64)
            self.buffer[-1] = 1
            
        elif self.mode == "mpc":
            self.vectors = crypten.cryptensor(torch.zeros((size+1,feature_size)))
            self.weights = crypten.cryptensor(torch.zeros(size+1))
            self.buffer = crypten.cryptensor(torch.zeros(size+1))
            self.buffer[-1] = 1
            self.IDX = crypten.cryptensor(torch.arange(1,size+2))
            self.IDX[-1] = 1

            self.lr = crypten.cryptensor(self.lr)
        if kernel_func == "rbf":
            self.kernel = self.kernel_rbf
    def kernel_rbf(self, x1, x2):
        gamma = 1.0
        if self.mode == "normal":
            return torch.exp(-gamma*(x1-x2)*(x1-x2))
        elif self.mode == "mpc":
            return (-gamma * (x1-x2) * (x1-x2)).exp()
    
    def predict(self,X_test):
        return self.custom_dot(self.weights,self.kernel(X_test,self.vectors))+self.beta
    def custom_sign(self,a):
        if self.mode=="normal":
            return torch.sign(a)
        else:
            return a.sign()
    def custom_min(self,a,b):
        if self.mode=="normal":
            return min(a,b)
        elif self.mode=="mpc":
            return (a>b)*b + (b>=a)*a
    def custom_argmin(self,x):
        if self.mode == "mpc":
            return x.argmin()
        else:
            return torch.nn.functional.one_hot(x.argmin(), num_classes=len(x))
    def custom_max(self,a,b):
        if self.mode=="normal":
            return max(a,b)
        elif self.mode=="mpc":
            return (a>b)*a + (b>=a)*b
    def custom_abs(self,a):
        if self.mode=="normal":
            return torch.abs(a)
        elif self.mode=="mpc":
            return a.abs()
    def custom_dot(self,a,b):
        if self.mode=="normal":
            return a.matmul(b)#a.dot(b)
        elif self.mode=="mpc":
            return a.matmul(b)

    def fit(self,X,y):
        size = X.shape[0]
        for it in tqdm(range(self.max_iters)):  
            idx = random.randint(1, size)
            pred_start = time.time()
            pred = self.predict(X[idx-1])
            pred_time = time.time()-pred_start
            weight_update_start = time.time()
            loss = self.custom_abs(y[idx-1]-pred)
            
            if self.mode == "mpc":
                gap = 1-(loss-self.eps)._ltz()
                
            else:
                gap = int(loss>self.eps)
            

            in_set = (self.IDX == idx)
            if self.mode == "normal":
                in_set = in_set.int()
            b_in_set = self.custom_dot(in_set,in_set)
            if self.mode == "normal":
                b_in_set = int(b_in_set)
            grad = gap*self.lr*self.custom_sign(pred-y[idx-1])
            edit_idx = ((in_set + (1-b_in_set)*self.buffer)>0)
            if self.mode == "normal":
                edit_idx = (edit_idx).int()
            to_edit = gap*edit_idx
            # update the indices, weights, and vectors
            self.weights = self.weights-grad*edit_idx
            self.beta = self.beta - gap*self.lr*self.custom_sign(pred-y[idx-1])
            weight_update_time = time.time()-weight_update_start

            if it < self.weights.shape[0]-1:
                min_idx = self.weights*0
                min_idx[it] = 1
            # we find the minimum weight, swap it with the buffer position 
            else:
                argmin_start = time.time()
                min_idx = self.custom_argmin(self.weights.abs())
                argmin_time = time.time()-argmin_start
                if self.verbose:
                    print("argmin time:",argmin_time)
            swap_start = time.time()
            self.IDX = gap*min_idx*(1+b_in_set)*(idx-self.IDX)+self.IDX
            self.weights = gap*min_idx*(1+b_in_set)*(self.weights[-1]-self.weights)+self.weights
            self.vectors = gap*min_idx.unsqueeze(1)*(1+b_in_set)*(X[idx-1]-self.vectors)+self.vectors
            
            swap_time = time.time()-swap_start
            # reset the buffer position:
            self.weights[-1] *= 0
            self.IDX[-1] *= 0
            self.vectors[-1] *= 0
            if self.verbose:
                print("pred time:",pred_time,"weight update time:",weight_update_time,"swap time:",swap_time)    




if __name__=="__main__":
    import time
    
    import matplotlib.pyplot as plt
    import sys
    from utils import mackey_glass
    from libsvm.svmutil import svm_train, svm_predict
    
    def train(features, labels, iters=None, budget=None, sigma=None, tol=None):
        model = svm_train(labels, features, '-e 1e-6 -s 3 -p 1e-5')
        return model

    def test(features, model, labels=None):
        if labels is None:
            labels = np.zeros(features.shape[0])
        pred, _, _ = svm_predict(labels, features, model)
        return pred
    crypten.init()
    total = int(sys.argv[1])
    size= int(total*0.8)
    test_size = total-size
    x = 190*(-0.5+torch.rand(size))#[1,1.1,1.2,1.3])
    # x = torch.from_numpy(x).float()
    def f(a):
        return np.sin(a*0.15)/a#a*2.4+1.3# #np.sin(a)#np.sin(a*0.15)/a#
    
    y = f(x)+ np.random.normal(0, 0.01, x.shape[0])#np.sin(x*1.1)+ np.random.normal(0, 0.1, x.shape[0])
    # y = mackey_glass(size,a=0.2,b=0.1,tau=17,sample=1.0)
    
    
    bsg = BSGD(0.5,15*size,learning_rate=0.005,margin=1e-3,dataset_size=size,mode="normal")
    # svr = SVR(100,C=10,eps=0.01,tol=1e-3,mode="normal")
    # print(bsg.IDX)
    # print(x.shape)
    t = time.time()
    svr = train(x.view(-1,1).numpy(),y.numpy())
    bsg.fit(x,y)
    
    print("Train time {:.4f}ms".format(time.time()-t))
    # print(bsg.IDX,bsg.weights)
    
    X_test = 190*(-0.5+np.random.rand(test_size))
    X_test.sort()
    X_test = torch.from_numpy(X_test).float()
    
    t = time.time()
    y_pred = bsg.predict(X_test)
    y_svr = test(X_test.view(-1,1).numpy(),svr,None)
    print("Predict time:",time.time()-t)
    plt.scatter(x, y,marker='+', color='black', label='data')
    plt.plot(X_test, y_svr, color='red', linewidth=3, label='SVR-SMO')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='SVR-BSGD')
    
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('sin(x*0.15)/x')
    plt.legend()
    plt.savefig("test.jpg")
