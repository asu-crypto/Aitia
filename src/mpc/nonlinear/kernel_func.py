import crypten, torch
import numpy as np

def polynomial_kernel(a, b, d=2):
    base = (a * b + 1).sum(dim=0)
    res = base
    for _ in range(1, d):
       res = res * base 
    return res

def RBF_kernel(x, a, gamma=0.1):
    # result = (-gamma * ((x - a) * (x - a)).sum(dim=1)).exp()
    # print(x.shape, a.shape)
    result = (-gamma * ((x - a) * (x - a))).exp()
    return result

if __name__ == '__main__':
    crypten.init()
    from crypten.config import cfg
    cfg.functions.exp_iterations = 6
    x = torch.Tensor([0.1])
    a = torch.Tensor([0.05, 0.01, 0.03])
    x_enc = crypten.cryptensor(x)
    a_enc = crypten.cryptensor(a)

    print(RBF_kernel(x_enc[0], a_enc, 1).get_plain_text())
    print(RBF_kernel(x, a, 1))