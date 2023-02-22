from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import gc

calssific_loss_weight=5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class my_loss1(nn.Module):
    def __init__(self):
        super(my_loss1, self).__init__()
        self.criteria1 = torch.nn.CrossEntropyLoss()
        self.criteria2 = torch.nn.MSELoss()
        self.b_xent = nn.BCEWithLogitsLoss()

    def forward(self, X, target, input1, X_AE1, input2, X_AE2):
        loss = calssific_loss_weight * self.criteria1(X, target)+ \
               self.criteria2(input1.float(), X_AE1) + \
               self.criteria2(input2.float(), X_AE2)

        return loss



def LL(new_feature, ZERO_ratio):
    ZERO_ratio = 0.2
    new_feature = torch.Tensor(new_feature)
    n_number = new_feature.size(0) * new_feature.size(1)
    n_zero = int(abs(ZERO_ratio * n_number))
    n_one = n_number - n_zero
    ze = torch.zeros(n_zero, 1)
    on = torch.ones(n_one, 1)
    L = torch.cat((ze, on), dim=0)
    id = np.arange(n_number)  # 得到index
    id = np.random.permutation(id)  #
    L = L[id]
    L = torch.reshape(L, [new_feature.size(0), new_feature.size(1)])
    return L

