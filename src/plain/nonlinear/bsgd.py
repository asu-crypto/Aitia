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

""" SGD SVR. """

import argparse
from data_processing import load_data
from functools import partial
import numpy as np
import os
import random
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from config import check_valid_pair, sigma_map, tol_map

INF = float('inf')

def rbf_kernel(x, y, gamma=0.1):
    return np.exp(-gamma * np.square(x - y).sum())

def predict(feature, model, kernel=rbf_kernel):
    """ Given a SVR model and a feature, return the prediction. """
    return np.sum([coef * kernel(feature, sp_vec) for sp_vec, coef in model.values()])

def train(features, labels, iters=10000, tol=0.1, budget=500, sigma=0.1):
    """ Train a SVR model using budgeted SGD. """
    H = {}
    size = len(labels)
    feature_size = len(features[0])
    kernel = partial(rbf_kernel, gamma=1./feature_size)
    import time
    # losses = []
    for t in range(1, iters+1):
        # t = time.time()
        idx = random.randint(0, size-1)
        pred = predict(features[idx], H, kernel)
        if np.abs(pred - labels[idx]) > tol:
            if idx in H.keys():
                if pred > labels[idx]:
                    H[idx][1] -= sigma # / np.sqrt(t)
                else:
                    H[idx][1] += sigma # / np.sqrt(t)
            else:
                if pred > labels[idx]:
                    H[idx] = [features[idx], -sigma] # / np.sqrt(t)]
                else:
                    H[idx] = [features[idx], +sigma] # / np.sqrt(t)]

                if len(H) > budget:
                    min_idx = None
                    min_val = INF
                    for (idx, (_, v)) in H.items():
                        if np.abs(v) < min_val:
                            min_idx = idx
                            min_val = np.abs(v)

                    H.pop(min_idx)
        # if t % 100 == 0:
        #     pred = test(test_features.reshape([-1,1]), H)-test_labels
        #     # pred = [(predict(test_features[i], H, kernel) - test_labels[i]) for i in range(len(test_labels))]
        #     # print("iter: {}, loss: {}".format(t, np.mean(np.square(pred))))
        #     losses.append(np.mean(np.square(pred)))

        # print("train finished in {} s".format(time.time()-t))
    # plt.plot(range(0,int(iters/100)*100,100),losses)
    # plt.xlabel('iterations')
    # plt.ylabel('mse_loss')
    # plt.title('mse_loss vs iterations, bsgd')
    # plt.savefig('loss.png')
    return H

def test(features, model):
    kernel = partial(rbf_kernel, gamma=1./len(features[0]))
    return [predict(feature, model, kernel) for feature in features]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify dataset abbreviation.')
    parser.add_argument('--budget_rate', type=float, default=1.)
    parser.add_argument('--data', default='abalone')
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--repetition', type=int, default=10)
    parser.add_argument('--target', type=int, default=0)
    args = parser.parse_args()

    if check_valid_pair(args.data, args.feature, args.target) is False:
        raise ValueError('Invalid feature-target pair.')

    train_data, test_data = load_data(args.data)
    train_feature = train_data[:, args.feature].reshape([-1, 1])
    train_target = train_data[:, args.target]
    test_feature = test_data[:, args.feature].reshape([-1, 1])
    test_target = test_data[:, args.target]
    iters = max(3000, 2*len(train_target))
    budget = int(args.budget_rate * len(train_target))
    # print(train_feature.min(), train_feature.max(), train_target.min(), train_target.max())
    # if args.budget is None:
    # 	args.budget = len(train_target)
    MSE_list = []
    for i in range(args.repetition):
        model = train(train_feature, train_target, iters, budget=budget, sigma=sigma_map[args.data], tol=tol_map[args.data])
        pred = test(test_feature, model)
        MSE_list.append(np.square(pred - test_target).mean())
    # print(np.mean(MSE_list), np.std(MSE_list))

    file_name = f'./results/{args.data}_{args.feature}_{args.target}_{args.budget_rate}_{args.repetition}.txt'
    txt_file = open(file_name, 'w')
    txt_file.write('%f, \t%f\n'%(np.mean(MSE_list), np.std(MSE_list)))
