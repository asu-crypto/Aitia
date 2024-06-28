import crypten
import torch

def gaussian_score_plaintext(x, y):
    return torch.log(torch.var(x)) + torch.log(torch.var(y))

# def gaussian_score(x, y):
#     result = x.var().log(input_in_01=True) + y.var().log(input_in_01=True)
#     return result

def gaussian_score(x, y):
    result = x.var() * y.var()
    return result

if __name__ == '__main__':
    crypten.init()
    from crypten.config import cfg
    cfg.functions.exp_iterations = 8
    cfg.functions.log_iterations = 16
    x = torch.Tensor([0.1, 1.0, 1.0])
    y = torch.Tensor([0.5, 0.1, 0.5])
    x_enc = crypten.cryptensor(x)
    y_enc = crypten.cryptensor(y)
    z_enc = x_enc - y_enc
    print(z_enc.get_plain_text())
    print(gaussian_score_plaintext(x, y))
    z_enc = gaussian_score(x_enc, y_enc)
    print(z_enc.get_plain_text())