import crypten
import numpy as np
import torch
import tqdm
from crypten.config import cfg
# cfg.functions.reciprocal_method = "log"
# cfg.encoder.precision_bits = 24
# cfg.functions.reciprocal_log_iters = 
# cfg.functions.exp_iterations= 16
# cfg.functions.reciprocal_initial = 1
def gaussian_elimination_inverse(matrix, I):
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square for inversion.")

    n = matrix.shape[0]
    augmented_matrix = np.hstack([matrix, I])

    # Perform Gaussian elimination
    for i in range(n):
        # Make the diagonal element 1
        diagonal_element = augmented_matrix[i, i]
        augmented_matrix[i, :] /= diagonal_element

        # Eliminate other elements in the column
        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j, :] -= factor * augmented_matrix[i, :]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = augmented_matrix[:, n:]

    return inverse_matrix


def invert_matrix(A, I, tol=None):
    n = A.shape[0]
    AM = A.clone()
    IM = I.clone()
    # AM = AM#*100
    # print(AM.shape)
    indices = list(range(n))
    for fd in tqdm.tqdm(range(n)):
        fdScaler = A[fd, fd].reciprocal()
        AM[fd, :] *=fdScaler#/= AM[fd,fd].clone()#fdScaler
        IM[fd, :] *=fdScaler#/= AM[fd,fd]#fdScaler
        # fdScaler = 1.0 / AM[fd, fd]
        # print(AM[fd,fd].get_plain_text().abs().max())
        
        # if AM[fd,fd] < 0.001:
        # scaler = crypten.cryptensor(1.0/AM[fd,fd].get_plain_text())#(AM[fd,fd]).reciprocal()#1.0/AM[fd,fd]#
        # scaler *= 10
        # print(scaler-1.0/AM[fd,fd])#.get_plain_text())
        # AM[fd, :] *= fdScaler
        # IM[fd, :] *= fdScaler
        # print(AM[fd,fd].get_plain_text())
        for i in indices:#[0:fd] + indices[fd+1:]:
            if i != fd:
                crScaler = AM[i, fd].clone()
                IM[i, :] -= crScaler * IM[fd, :]
                AM[i, :] -= crScaler * AM[fd, :]
    # print(A.matmul(IM).get_plain_text())
    return IM#*100

def RBF_kernel(x, a, gamma=1):
    result = (-gamma * ((x - a) * (x - a))).exp()
    return result

def train_gp_mpc(args):
    # crypten.init()
    (features_enc, labels_enc, test_features, test_labels, I) = args
    delta = 1e-4
    feat = features_enc.expand(-1, features_enc.shape[0])
    feat_new = feat.transpose(0, 1)
    K_train = RBF_kernel(feat, feat_new) + delta * I
    
    #.float()
    # print(K_test)
    # for fd in range(K_train.shape[0]):
        # print(K_train[fd,fd].get_plain_text())
    K_train_inv = invert_matrix(K_train,I)#.get_plain_text()#, I.get_plain_text())#.get_plain_text()
    # K_train_inv = torch.from_numpy(np.linalg.inv(K_train.get_plain_text().numpy())).float()
    # K_train_inv = crypten.cryptensor(K_train_inv)
    # K_train_inv = torch.from_numpy(np.linalg.inv(K_train.numpy())).float()
    # K_train_inv = torch.from_numpy(np.linalg.inv(K_train.get_plain_text().numpy())).float()
    import time
    t = time.time()
    for _ in range(1):
        feat_new_ = features_enc.expand(-1, test_features.shape[0]).transpose(0, 1)
        feat_test = test_features.expand(-1, features_enc.shape[0])
        K_test = RBF_kernel(feat_test, feat_new_)
        res = K_test.matmul(K_train_inv)
        res = res.matmul(labels_enc)

    print("test time:",(time.time()-t)/100)
    return res

if __name__ == '__main__':

    crypten.init()
    A = np.random.rand(50, 50)
    # A[1,1]/=10000
    # A[3,3]/=1000
    I_enc = crypten.cryptensor(torch.eye(A.shape[0]).float())
    I = torch.eye(A.shape[0]).float()
    A_torch = torch.from_numpy(A).float()
    A_enc = crypten.cryptensor(A_torch)
    np_res = np.linalg.inv(A)
    my_res = invert_matrix(A_enc, I_enc)
    # my_res = invert_matrix(A_torch, I)#.numpy()
    # print(my_res.get_plain_text())
    print(my_res.get_plain_text().matmul(A_torch))#A_enc.get_plain_text()))
    # print(my_res.matmul(A_torch))#.get_plain_text())
    # print(np.sqrt(np.linalg.norm(np_res - my_res) / A.shape[0]**2))
    # print(np.sqrt(np.linalg.norm(np_res) / A.shape[0]**2))
