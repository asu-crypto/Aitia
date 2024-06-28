import numpy as np
import random
import crypten
from tqdm import tqdm
from crypten.config import cfg
import crypten.communicator as comm
import torch
import time
# from dependence_score import gaussian_score
def gaussian_score(x, y):
    return np.log(np.var(x)) + np.log(np.var(y))
class SVR():
    def __init__(self, max_iter=1000,C=1.0,eps=0.1,tol=1e-3,mode="mpc"):
        self.max_iter = max_iter
        self.C = C #C
        self.epsilon = eps #epsilon
        self.tol = tol # tolerance
        self.mode=mode
        self.kernel = self.kernel_rbf
    
    def kernel_poly(self,a, b, d=2):
        # print(b.transpose(-1,0).shape,b.shape,a.shape)#, np.matmul(a,b).shape)
        base = (np.dot(a,b) + 1)#.sum(0)#dim=0)
        res = base
        for _ in range(1, d):
            res = res * base 
        # print(res.shape)
        return res
    def kernel_linear(self,x1,x2):
        if self.mode=="mpc":
            return x1.dot(x2.transpose(-1,0))
        else:
            return x1.dot(x2)
    def kernel_rbf(self, x1, x2):
        gamma = 1.0#1 / x1.shape[0]
        if self.mode == "normal":
            return np.exp(-gamma*(x1-x2)*(x1-x2))
        elif self.mode == "mpc":
            t = time.time()
            sq = (x1-x2)*(x1-x2)
            print("square time:",time.time()-t)
            t = time.time()
            prod = -gamma*sq
            print("prod time:",time.time()-t)
            t = time.time()
            # res = prod.exp()
            iters = cfg.functions.exp_iterations
            result = 1 + prod.div(2**iters)
            print("div time:",time.time()-t)
            s = result.shape[0]
            s2 = result.shape[1]
            for it in range(iters):
                t = time.time()
                for seg in range(s//20+1):
                    for seg2 in range(s2//20+1):
                        result[seg*20:seg*20+20,seg2*20:seg2*20+20] = result[seg*20:seg*20+20,seg2*20:seg2*20+20].square()
                print(f"square2 {it} time:",time.time()-t)
            return result
            # print(res.shape)
            # print("exp time:",time.time()-t)
            # return res
            # return (-gamma * (x1-x2) * (x1-x2)).exp()
    def predict(self,X):
        # if self.mode=="mpc":
        #     self.alphas = self.alphas.get_plain_text()
        #     self.b = self.b.get_plain_text()
            # self.mode="normal"
            # X = crypten.cryptensor(X)
        # res = 0
        # for i in range(len(self.X)):
        #     res += self.alphas[i]*self.kernel(X,self.X[i])
        # return res + self.b
        # print(self.X.shape)
        # print(self.X.shape, self.alphas.shape,X.shape)
        t= time.time()
        knl = self.kernel(X,self.X)
        print("kernel time:",time.time()-t)
        t= time.time()
        dot = self.custom_dot(self.alphas,knl)
        print("dot time:",time.time()-t)
        t = time.time()
        res = dot + self.b
        print("add time:",time.time()-t)
        return res
        # return self.custom_dot(self.alphas,self.kernel(X,self.X)) + self.b
        # return sum([self.alphas[i]*self.kernel(self.X[i],X) for i in range(len(self.X))]) + self.b

    def examine_example(self,i,y):
        num_changed = 0
        condition1 = (self.errors[i] < self.tol) * (self.alphas[i] < self.C)
        condition2 = (self.errors[i] > self.tol) * (self.alphas[i] > 0)
        total_condition = (condition1 + condition2)>0
        
        idx = (self.alphas*(self.alphas-self.C)!=0).sum()
        
        # assert False
        if self.mode == "mpc":
            j = (self.errors[i]>0)*self.errors.argmin() + (self.errors[i]<=0)*self.errors.argmax()
            j = j[0]
            j = i + (j-i)*(idx>1)*total_condition
        else:
            j = (self.errors[i]>0)*np.argmin(self.errors) + (self.errors[i]<=0)*np.argmax(self.errors)
            j = i + (j-i)*(idx>1)*total_condition
        
        if self.mode=="mpc":
            self.step(j.get_plain_text(),i,y,(j-i)!=0)
        else:
            self.step(int(j),int(i),y,(j-i)!=0)
        num_changed += (i!=j)
        # if num_changed > 1:
        #     print(num_changed)
        # count = 0 
        for k in np.random.permutation(np.arange(len(self.alphas))):
            cond_k = self.alphas[k]*(self.alphas[k]-self.C)!=0
            total_cond = (num_changed==0)*float(k!=i)*(cond_k)*total_condition
            # self.step(k,i,y,total_cond>0)
            num_changed += total_cond
            # if self.mode=="mpc":
            #     total_cond = total_cond.get_plain_text().int()
            if total_cond.get_plain_text() > 0:
                self.step(k,i,y,total_cond>0)
                # break
            # count += 1
            # if count == 2: #run at most 2 times
            #     break
        
        count = 0 
        for k in np.random.permutation(np.arange(len(self.alphas))): # loop  over all possible i1, starting at a random point
            # if (i!=k):
            total_cond = (i!=k)*(num_changed==0)*total_condition
            self.step(k,i,y,total_cond)
            num_changed += total_cond
            count += 1
            if count == 2: #run at most 2 times because we only need i!=k for 1 time and the range is over all possible values
                break
        return num_changed
    def step(self,idx1, idx2, y,cond):
        # cond = float(cond)
        cond = cond > 0
        a1 = self.alphas[idx1]
        a2 = self.alphas[idx2]
        y1 = y[idx1]
        y2 = y[idx2]
        f1 = self.predict(self.X[idx1,:])#self.custom_dot(self.alphas, self.knmt[idx1,:])#self.alphas.dot(self.knmt[idx1,:])
        f2 = self.predict(self.X[idx2,:]) #self.custom_dot(self.alphas, self.knmt[idx2,:]) #self.alphas.dot(self.knmt[idx2,:])
        # print("error:",(f1-y1).get_plain_text()**2+(f2-y2).get_plain_text()**2)
        
        s = a1 + a2
        eta = self.knmt[idx1,idx1] + self.knmt[idx2,idx2] - 2*self.knmt[idx1,idx2]+1e-8
        # print(eta)
        if self.mode == "mpc":
            eta_inv = eta.reciprocal()#crypten.cryptensor(1.0/eta.get_plain_text())#eta.reciprocal()#
        elif self.mode=="normal":
            eta_inv = 1/eta
        delta = 2*self.epsilon * eta_inv
        
        self.alphas[idx1] = (a1 + eta_inv * (y1-y2-f1+f2))*cond + (1-cond)*self.alphas[idx1]
        # print("idx1:",self.alphas[idx1].get_plain_text(),cond.get_plain_text())
        self.alphas[idx2] = (s - self.alphas[idx1])*cond + (1-cond)*self.alphas[idx2]
        # print("idx2:",self.alphas[idx2].get_plain_text())
        parent_condition = (self.alphas[idx1]*self.alphas[idx2] < 0)
        child_condition1 = (self.custom_abs(self.alphas[idx1]) >= delta)
        child_condition2 = (self.custom_abs(self.alphas[idx2]) >= delta)
        child_condition = child_condition1 * child_condition2
        res1 = self.alphas[idx1] - delta*self.custom_sign(self.alphas[idx1])#sgn(alpha(i1))*delta
        res2 = self._step(self.custom_abs(self.alphas[idx1])-self.custom_abs(self.alphas[idx2]))*s
        self.alphas[idx1] = cond*parent_condition * (child_condition*res1 + (1-child_condition)*res2) + (1-cond)*self.alphas[idx1]
        
        L = self.custom_max(s - self.C, -self.C)
        H = self.custom_min(self.C, s + self.C)
        self.alphas[idx1] = (1-cond)*self.alphas[idx1]+cond*self.custom_min(self.custom_max(self.alphas[idx1],L),H)
        self.alphas[idx2] = (1-cond)*self.alphas[idx2]+ cond*(s - self.alphas[idx1])
        delta_error = (self.alphas[idx1]-a1)*self.knmt[idx1,:] + (self.alphas[idx2]-a2)*self.knmt[idx2,:]
        delta_error = cond*delta_error
        
        self.errors = self.errors + delta_error
        
            
    def custom_sign(self,a):
        if self.mode=="normal":
            return np.sign(a)
        else:
            return a.sign()
    def custom_min(self,a,b):
        if self.mode=="normal":
            return min(a,b)
        elif self.mode=="mpc":
            return (a>b)*b + (b>=a)*a
    def custom_max(self,a,b):
        if self.mode=="normal":
            return max(a,b)
        elif self.mode=="mpc":
            return (a>b)*a + (b>=a)*b
    def custom_abs(self,a):
        if self.mode=="normal":
            return np.abs(a)
        elif self.mode=="mpc":
            return a.abs()
    def custom_dot(self,a,b):
        # print(a.shape,b.shape,a.dot(b).shape)
        if self.mode=="normal":
            return a.matmul(b)
        elif self.mode=="mpc":
            return a.matmul(b)

    def _step(self,x):
        if self.mode == "normal":
            s = 1
            if x < 0:
                s = 0
            return s
        else:
            return x>=0
    def fit(self,X,y,actual_fit = True):
        # X = crypten.cryptensor(X)
        # y = crypten.cryptensor(y)
        # X = X.get_plain_text().numpy()
        # y = y.get_plain_text().numpy()
        y = y.squeeze()
        self.support_vectors = X 
        self.X = X#.squeeze()
        num_vectors = self.support_vectors.shape[0]
        self.alphas = torch.zeros((num_vectors,))
        self.b = 0
        self.errors = -y
        
        num_change = 0
        examine_all = True
        counter = 0
        if self.mode == "mpc":
            # X = crypten.cryptensor(X)
            # y = crypten.cryptensor(y)
            self.alphas = crypten.cryptensor(self.alphas)
            # self.b = crypten.cryptensor(self.b)
            # self.errors = crypten.cryptensor(self.errors)
            self.X = X#.squeeze()
        self.knmt = self.kernel(X,X.transpose(-1,0))
        if not actual_fit:
            return
        for counter in tqdm(range(self.max_iter)):#num_change > 0 or examine_all:
            num_change = 0
            if self.mode == "normal":
                alpha_old = np.copy(self.alphas)
            else:
                alpha_old = self.alphas.clone()#np.copy(self.alphas)
            if examine_all:
                
                for i in tqdm(range(num_vectors)):
                    # print("====Communication Stats====")
                    # print(comm.get())
                    # print("Rounds before iteration: {}".format(comm.get().comm_rounds))
                    # print("Bytes before iteration: {}".format(comm.get().comm_bytes))
                    # print("Communication time before iteration: {}".format(comm.get().comm_time))
                    num_changed = self.examine_example(i,y)
                    num_change += num_changed
                    # print("====Communication Stats====")
                    # print(comm.get())
                    # print("Rounds after iteration: {}".format(comm.get().comm_rounds))
                    # print("Bytes after iteration: {}".format(comm.get().comm_bytes))
                    # print("Communication time after iteration: {}".format(comm.get().comm_time))
                    if i == 10:
                        break
            else:
                for i in tqdm(range(num_vectors)):
                    num_change += (self.alphas[i]!=0)*(self.alphas[i]!=self.C)*self.examine_example(i,y)
                    if i==10:
                        break
                    #if (self.alphas[i]!=0) and (self.alphas[i]!=self.C):
                    #    num_change += self.examine_example(i,y)
            examine_all = not examine_all
            if self.mode=="normal":
                change = np.linalg.norm(self.alphas-alpha_old)
            else:
                change = np.linalg.norm(self.alphas.get_plain_text()-alpha_old.get_plain_text())
            if change < self.tol:
                print("Converged after {} steps".format(counter+1))
                break
        self.b = -self.errors.mean()
        # print(self.alphas, self.b)
        if self.mode == "mpc":
            print(self.alphas.get_plain_text(), self.b.get_plain_text())
        print("Converged after {} steps".format(counter))
            # print(self.errors)
            # break
        #pass

if __name__=="__main__":
    import argparse
    import torch
    import pandas as pd
    crypten.init()
    cfg.encoder.precision_bits = 18
    # Example usage:
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.sort(5 * np.random.rand(100, 1), axis=0)# np.arange(0,5,5/40)[:,np.newaxis]
    # print(X_train.shape)
    y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0]) #np.sin(X_train).ravel()
    
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    # print(X_test)
    # X_train = torch.tensor
    # X_train = torch.from_numpy(X_train).float()#,dtype="float")
    # y_train = torch.from_numpy(y_train).float()
    # X_test = torch.from_numpy(X_test).float()
    # X_train = crypten.cryptensor(X_train)
    # y_train = crypten.cryptensor(y_train)
    # X_test= crypten.cryptensor(X_test)
    # Fit SVRSMO model
    svr_smo = SVR(C=1.0, tol=1e-7,eps=1e-5, max_iter=3,mode="normal")

    svr_smo.fit(X_train, y_train)

    # Predict
    y_pred = svr_smo.predict(X_test)
    try:
        X_train=X_train.get_plain_text()
        y_train=y_train.get_plain_text()
        X_test=X_test.get_plain_text()
        
        y_pred = y_pred.get_plain_text()
    except:
        pass
    # print(y_pred)
    # Plot results
    # print(X_test,y_pred)
    import matplotlib.pyplot as plt

    plt.scatter(X_train, y_train, color='black', label='data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='SVR-SMO')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression (SVR) with SMO')
    plt.legend()
    plt.savefig("test.jpg")

    # # import crypten
    # # from utils import A2B, AND
    
    # x = crypten.cryptensor([1,2,3])
    # y = crypten.cryptensor([2,3,4])
    # z = svr_smo.custom_dot(x,y,"mpc")
    # # z = AND(A2B(x>2) , A2B(y>3))#svr_smo.custom_dot(x,y,"mpc")
    # print(z.get_plain_text())
    if False:
        parser = argparse.ArgumentParser()
        parser.add_argument('--C', type=float, default=1)
        parser.add_argument('--eps', type=float, default=0.1)
        parser.add_argument('--kernel', default='RBF')
        parser.add_argument('--gamma', type=float, default=-1.0)
        parser.add_argument('--tau', type=float, default=1e-12)
        parser.add_argument('--threshold', type=int, default=100)
        parser.add_argument('--checkpoint', type=int, default=10)
        parser.add_argument('--feature',type=int,default=2)
        parser.add_argument('--target',type=int,default=5)
        args = parser.parse_args()
        name = "liver_disorder"
        train = pd.read_csv('../../../data/'+name+'_train')
        test = pd.read_csv('../../../data/'+name+'_test')
        train_data= train.iloc[:, 1:].to_numpy()
        test_data = test.iloc[:, 1:].to_numpy()
        # return train_data, test_data
        train_features = train_data[:, args.feature:args.feature+1]
        train_labels = train_data[:, args.target:args.target+1]
        test_features = test_data[:, args.feature:args.feature+1]
        test_labels = test_data[:, args.target:args.target+1]
        
        svr_smo.fit(train_labels,train_features)
        print((svr_smo.alphas>0).sum())
        res_x = svr_smo.predict(test_labels)-test_features.squeeze()
        svr_smo = SVR(C=1.0, tol=1e-7,eps=1e-5, max_iter=5)
        svr_smo.fit(train_features,train_labels)
        # print(svr_smo.alphas)
        res_y = svr_smo.predict(test_features) - test_labels.squeeze()
        print('mse:', np.mean(np.square(res_y)),np.mean(np.square(res_x)))
        dep_xy = gaussian_score(train_features, res_y)
        dep_yx = gaussian_score(test_labels, res_x)
        print(dep_xy,dep_yx)
        if dep_xy >= dep_yx:
            print('Y->X')
            # res_list.append('Y->X')
        else:
            print('X->Y')