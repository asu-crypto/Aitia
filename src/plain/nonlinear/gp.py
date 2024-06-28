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

""" Gaussian process regression. """

import argparse
from data_processing import load_data
import numpy as np
import os
import sklearn.gaussian_process as gp
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from data_processing import load_data
from config import check_valid_pair

def train(features, labels, iters=None, budget=None, sigma=None, tol=None):
    """ TODO """

    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    model.fit(features, labels)

    return model

def test(features, model):
    """ TODO """

    pred, std = model.predict(features, return_std=True)
    return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Specify dataset abbreviation.')
    parser.add_argument('--data', default='abalone')
    parser.add_argument('--feature', type=int)
    parser.add_argument('--target', type=int)
    args = parser.parse_args()

    if check_valid_pair(args.data, args.feature, args.target) is False:
        raise ValueError('Invalid feature-target pair.')

    train_data, test_data = load_data(args.data)
    train_feature = train_data[:, args.feature].reshape([-1, 1])
    train_target = train_data[:, args.target]
    test_feature = test_data[:, args.feature].reshape([-1, 1])
    test_target = test_data[:, args.target]
    model = train(train_feature, train_target)
    pred = test(test_feature, model)
    MSE = np.square(pred-test_target).mean()
    print("MSE:", MSE)
