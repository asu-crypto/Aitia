import time
import argparse
import crypten
from crypten.config import cfg
import crypten.communicator as comm
import torch
#ignore warnings
import warnings;
warnings.filterwarnings("ignore")
from nonlinear.smo import SVR
from nonlinear.utils import load_data
from dependence_score import gaussian_score
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from nonlinear.gp import train_gp_mpc
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from config import sigma_map, tol_map
from mpc.nonlinear.bsgd import BSGD
from multiprocess_launcher import MultiProcessLauncher


# function to train using GP Regression
def train_mp_gp(args, train_features, train_labels, test_features, test_labels):
    crypten.init()
    cfg.communicator.verbose = True
    cfg.encoder.precision_bits = 18
    features_enc = crypten.cryptensor(train_features)
    labels_enc = crypten.cryptensor(train_labels)
    start = time.time()
    test_features = crypten.cryptensor(test_features)
    test_labels = crypten.cryptensor(test_labels)
    

    I1 = crypten.cryptensor(torch.eye(features_enc.shape[0]).float())
    I2 = crypten.cryptensor(torch.eye(labels_enc.shape[0]).float())
    parallel_args = [(features_enc, labels_enc, test_features, test_labels, I1), (labels_enc, features_enc, test_labels, test_features, I2)]
    gp_res = [train_gp_mpc(parallel_args[0]),]
    
    end = time.time()
    res_y = gp_res[0] - test_labels
    dep_xy = gaussian_score(test_features, res_y)

    dep_xy = dep_xy.get_plain_text()

    print("====Communication Stats====")
    print(comm.get())
    print("Rounds: {}".format(comm.get().comm_rounds))
    print("Bytes: {}".format(comm.get().comm_bytes))
    print("Communication time: {}".format(comm.get().comm_time))
    print('mse:', np.linalg.norm(res_y.get_plain_text().numpy()) / res_y.shape[0])
    print(dep_xy, end-start)

# function to train using SMO SVR
def train_mp_smo(args,train_features,train_labels,test_features,test_labels):
    crypten.init()
    cfg.communicator.verbose=True
    gamma = 1. / train_features.shape[1]
    features_enc = crypten.cryptensor(train_features)
    labels_enc = crypten.cryptensor(train_labels)
    test_features = crypten.cryptensor(test_features).squeeze()
    test_labels = crypten.cryptensor(test_labels).squeeze()
    train_start = time.time()
    svr = SVR(max_iter=2,C=1.0,eps=0.1,tol=tol_map[args.dataset],mode="mpc")
    svr.fit(features_enc,labels_enc, True)
    
    train_end = time.time()
    test_start = time.time()
    for _ in range(1):
        res_y = svr.predict(test_features) - test_labels#.squeeze()
    test_end = time.time()
    dep_xy = gaussian_score(test_features,res_y)
    # dep_xy = dep_xy.get_plain_text()
    print("====Communication Stats====")
    print(comm.get())
    print("Rounds: {}".format(comm.get().comm_rounds))
    print("Bytes: {}".format(comm.get().comm_bytes))
    print("Communication time: {}".format(comm.get().comm_time))
    print('mse:', np.mean(np.square(res_y.get_plain_text().numpy())))
    print('dep:', dep_xy, dep_xy.get_plain_text(), 'time:', train_end-train_start, "test time:", (test_end-test_start)/1)

# function to train using BSGD-SVR
def train_mp_bsgd(args, train_features, train_labels, test_features, test_labels):
    crypten.init()
    cfg.communicator.verbose = True
    train_features = train_features.squeeze()
    train_labels = train_labels.squeeze()
    features_enc = crypten.cryptensor(train_features)
    labels_enc = crypten.cryptensor(train_labels)
    test_features = crypten.cryptensor(test_features).squeeze()
    test_labels = crypten.cryptensor(test_labels).squeeze()
    size = train_features.shape[0]
    bsgd = BSGD(budget_size=args.budget_rate, max_iters=int(2*size), dataset_size=size, learning_rate=0.01, margin=1e-4, kernel_func = "rbf",mode="mpc",verbose=args.verbose)
    
    start = time.time()
    bsgd.fit(features_enc,labels_enc)
    end = time.time()
    print("====Communication Stats====")
    print(comm.get())
    print("Rounds before test: {}".format(comm.get().comm_rounds))
    print("Bytes before test: {}".format(comm.get().comm_bytes))
    print("Communication time before test: {}".format(comm.get().comm_time))
    pred_result = bsgd.predict(test_features.squeeze())
    test_start = time.time()
    
    for _ in range(1):
        res_y = bsgd.predict(test_features) - test_labels
    test_end = time.time()
    dep_xy = gaussian_score(test_features, res_y)
    
    print("Rounds: {}".format(comm.get().comm_rounds))
    print("Bytes: {}".format(comm.get().comm_bytes))
    print("Communication time: {}".format(comm.get().comm_time))
    print('mse:', np.mean(np.square(res_y.get_plain_text().numpy())))
    print('dep:', dep_xy, dep_xy.get_plain_text(), 'time:', end-start, "test time:", test_end-test_start)

if __name__ == '__main__':
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', default='bsgd', help='Algorithm to test: bsgd gp.')
    parser.add_argument('--budget_rate', type=float, default=0.5, help='Budget of support vectors.')
    parser.add_argument('--dataset', default='abalone', help='Name of the dataset.')
    parser.add_argument('--epoch', type=int, default=2, help='Number of passes through the training dataset.')
    parser.add_argument('--num-party',type=int,default=3)
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--verbose', default=False, help='Whether output debugging info?')
    parser.add_argument('--feature', type=int, default=7)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument("--test-direction",type=str, default="f2t",help="Direction of the test, either f2t (feature to target) or f2t(target to feature)")
   
    args = parser.parse_args()

    if args.dataset in ['abalone', 'arrhythmia', 'liver_disorder', 'income', 'ncep']:
        train_data, test_data = load_data(args.dataset)
        train_features = torch.from_numpy(train_data[:, args.feature:args.feature+1]).float()
        train_labels = torch.from_numpy(train_data[:, args.target:args.target+1]).float()
        test_features = torch.from_numpy(test_data[:, args.feature:args.feature+1]).float()
        test_labels = torch.from_numpy(test_data[:, args.target:args.target+1]).float()
    else:
        raise NotImplementedError(f'Data {args.dataset} is not supported.')

    budget = int(args.budget_rate * train_features.shape[0])
    gamma = 1. / train_features.shape[1]

    if args.alg == 'smo':
        launcher = MultiProcessLauncher(args.num_party, train_mp_smo, fn_args=[args,train_features,train_labels,test_features, test_labels])
        launcher.start()
        launcher.join()
        launcher.terminate()
    
    if args.alg == 'bsgd':
        launcher = MultiProcessLauncher(args.num_party, train_mp_bsgd, fn_args=[args,train_features,train_labels,test_features, test_labels])
        launcher.start()
        launcher.join()
        launcher.terminate()
        
    
    elif args.alg == 'gp':
        launcher = MultiProcessLauncher(args.num_party, train_mp_gp, fn_args=[args,train_features,train_labels,test_features, test_labels])
        launcher.start()
        launcher.join()
        launcher.terminate()
        