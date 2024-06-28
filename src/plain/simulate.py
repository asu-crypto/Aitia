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

""" Causal Inference. """

import argparse
from dependence_score import gaussian_score
import importlib
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from config import check_valid_pair, sigma_map, tol_map
from data_processing import load_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget_rate', type=float, default=0.5, help="Budget divided by the size of the training set")
    parser.add_argument('--alg', default='gp', help="Candidate algorithms: gp, smo, bsgd")
    parser.add_argument('--dataset', default='liver_disorder', help="Candidate datasets: abalone, arrhythmia, income, liver_disorder, ncep")
    parser.add_argument('--feature', type=int, default=0)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()

    if check_valid_pair(args.dataset, args.feature, args.target) is False:
        raise ValueError('Invalid feature-target pairs.')
    
    reg = importlib.import_module("nonlinear."+args.alg)

    train_data, test_data = load_data(args.dataset)
    train_x = train_data[:, args.feature]
    train_y = train_data[:, args.target]
    test_x = test_data[:, args.feature]
    test_y = test_data[:, args.target]

    iters = int(2*len(train_x))#100#args.iters#2*len(train_x)
    budget = int(args.budget_rate *  len(train_x))
    print("Budget: ", budget)

    file_name = f'./results/%s_%s_%d_%d_%.1f.txt'%(args.dataset, args.alg, args.feature, args.target, args.budget_rate)
    txt_file = open(file_name, 'w')
    txt_file.write('budget, xymean, xystd, yxmean, yxstd\n')

    xy_mse_list = []
    yx_mse_list = []
    res_list = []
    for _ in range(args.iters):
        model_xy = reg.train(train_x.reshape([-1, 1]), train_y, iters=iters, budget=budget, sigma=sigma_map[args.dataset], tol=tol_map[args.dataset])
        model_yx = reg.train(train_y.reshape([-1, 1]), train_x, iters=iters, budget=budget, sigma=sigma_map[args.dataset], tol=tol_map[args.dataset])
        
        res_x = reg.test(test_y.reshape([-1, 1]), model_yx) - test_x
        res_y = reg.test(test_x.reshape([-1, 1]), model_xy) - test_y
        xy_mse_list.append(np.mean(np.square(res_x)))
        yx_mse_list.append(np.mean(np.square(res_y)))
        print(np.mean(np.square(res_x)), np.mean(np.square(res_y)))
        dep_xy = gaussian_score(test_x, res_y)
        dep_yx = gaussian_score(test_y, res_x)
        print(dep_xy, dep_yx)
        if dep_xy >= dep_yx:
            print('Y->X')
            res_list.append('Y->X')
        else:
            print('X->Y')
            res_list.append('X->Y')
    print(xy_mse_list, yx_mse_list)
    txt_file.write(f'%f, %f, %f, %f, %f'%(args.budget_rate, np.mean(xy_mse_list), np.std(xy_mse_list), np.mean(yx_mse_list), np.std(yx_mse_list)))
