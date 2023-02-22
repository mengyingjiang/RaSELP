import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drug_encoding = 'CNN'
bert_n_heads=4
cov2KerSize=50
cov1KerSize=25

task = "task2"

epo_num = 120
if task == "task1":
    drop_out_rating = 0.3
    learn_rating = 0.00002
    batch_size = 512
    graph_dim = 1024
    n_hop = 0
    dim_AE = 1000

if task == "task2":
    drop_out_rating = 0.2
    learn_rating = 0.000005
    batch_size =512
    graph_dim =1024
    n_hop = 3
    dim_AE = 1500

if task == "task3":
    drop_out_rating = 0.2
    learn_rating = 0.000005
    batch_size =512  #
    graph_dim =1024
    n_hop = 3
    dim_AE = 1500


class CNN_concat(nn.Sequential):
    def __init__(self, out_dim, encoding,  **config):
        super(CNN_concat, self).__init__()
        if encoding == 'drug':
            in_ch = [64] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((64, 200))
            # n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, out_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size,dim):
        super(AE1, self).__init__()
        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, (self.vector_size + dim) // 2)
        self.bn1 = torch.nn.BatchNorm1d((self.vector_size + dim) // 2)

        self.l2 = torch.nn.Linear((self.vector_size + dim) // 2, dim)

        self.att2s = torch.nn.ModuleList(
            [EncoderLayer((self.vector_size + dim) // 2, bert_n_heads) for _ in range(2)])

        self.l3 = torch.nn.Linear(dim, (self.vector_size + dim) // 2)
        self.bn3 = torch.nn.BatchNorm1d((self.vector_size + dim) // 2)

        self.l4 = torch.nn.Linear((self.vector_size + dim) // 2, self.vector_size)

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.ac = gelu

    def forward(self, X):
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        for att2 in self.att2s:
            X = att2(X)

        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, X_AE

class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):  # [1966,4]
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)   # [1966,4]
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)  # 多头注意力去聚合其他DDI的特征
        X = self.AN1(output + X)  # 残差连接+LayerNorm

        output = self.l1(X)  # FC
        X = self.AN2(output + X)  # 残差连接+LayerNorm

        return X

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):   # [1966,4]

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads  # 491.5
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim  #1966
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)  # [1966,491]*4
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)  # [1966,491]*4
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)  # [1966,491]*4

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)  # [256,1966]->[256,1964]->[256,4,491]->[4,256,625]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # [4,256,491]*[4,491,256]->[4,256,256]
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        scores = torch.nn.Softmax(dim=-1)(scores)
        scores = torch.matmul(scores, V)  # [4,256,256]*[4,256,625]->[4,256,491]
        # context: [len_q, n_heads * d_v]
        scores = scores.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        scores = self.fc(scores)
        return scores

class dotproduct(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features):
        super(dotproduct, self).__init__()
        self.in_features = in_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mul(input, self.weight)
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' ')'

class RGCN(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, N_relation, in_features, out_features, bias=True):
        super(RGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N_relation = N_relation
        self.weight = torch.nn.Parameter(torch.FloatTensor(N_relation, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = torch.matmul(adj, input)
        output = torch.matmul(input, self.weight)
        output = torch.sum(output, dim=0)
        # output = output.transpose(1, 2).reshape(-1, self.N_relation * self.out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HomoGCN(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, n_hop,bias=True):
        super(HomoGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_hop = n_hop
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        for i in range(self.n_hop):
            input = torch.mm(adj, input)

        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def generate_config(drug_encoding=None,
                    result_folder="./result/",
                    input_dim_drug=1024,
                    hidden_dim_drug=256,
                    cls_hidden_dims=[1024, 1024, 512],
                    mlp_hidden_dims_drug=[1024, 256, 64],
                    batch_size=256,
                    train_epoch=10,
                    test_every_X_epoch=20,
                    LR=1e-4,
                    transformer_emb_size_drug=128,
                    transformer_intermediate_size_drug=512,
                    transformer_num_attention_heads_drug=8,
                    transformer_n_layer_drug=8,
                    transformer_dropout_rate=0.1,
                    transformer_attention_probs_dropout=0.1,
                    transformer_hidden_dropout_rate=0.1,
                    mpnn_hidden_size=50,
                    mpnn_depth=3,
                    cnn_drug_filters=[32, 64, 96],
                    cnn_drug_kernels=[4, 6, 8],
                    rnn_Use_GRU_LSTM_drug='GRU',
                    rnn_drug_hid_dim=64,
                    rnn_drug_n_layers=2,
                    rnn_drug_bidirectional=True,
                    num_workers=0,
                    cuda_id=None,
                    ):
    base_config = {'input_dim_drug': input_dim_drug,
                   'hidden_dim_drug': hidden_dim_drug,  # hidden dim of drug
                   'cls_hidden_dims': cls_hidden_dims,  # decoder classifier dim 1
                   'batch_size': batch_size,
                   'train_epoch': train_epoch,
                   'test_every_X_epoch': test_every_X_epoch,
                   'LR': LR,
                   'drug_encoding': drug_encoding,
                   'result_folder': result_folder,
                   'binary': False,
                   'num_workers': num_workers,
                   'cuda_id': cuda_id
                   }
    if not os.path.exists(base_config['result_folder']):
        os.makedirs(base_config['result_folder'])
    if drug_encoding == 'Morgan':
        base_config['mlp_hidden_dims_drug'] = mlp_hidden_dims_drug  # MLP classifier dim 1
    elif drug_encoding == 'CNN':
        base_config['cnn_drug_filters'] = cnn_drug_filters
        base_config['cnn_drug_kernels'] = cnn_drug_kernels
    elif drug_encoding == 'CNN_RNN':
        base_config['rnn_Use_GRU_LSTM_drug'] = rnn_Use_GRU_LSTM_drug
        base_config['rnn_drug_hid_dim'] = rnn_drug_hid_dim
        base_config['rnn_drug_n_layers'] = rnn_drug_n_layers
        base_config['rnn_drug_bidirectional'] = rnn_drug_bidirectional
        base_config['cnn_drug_filters'] = cnn_drug_filters
        base_config['cnn_drug_kernels'] = cnn_drug_kernels
    elif drug_encoding == 'Transformer':
        base_config['input_dim_drug'] = 2586
        base_config['transformer_emb_size_drug'] = transformer_emb_size_drug
        base_config['transformer_num_attention_heads_drug'] = transformer_num_attention_heads_drug
        base_config['transformer_intermediate_size_drug'] = transformer_intermediate_size_drug
        base_config['transformer_n_layer_drug'] = transformer_n_layer_drug
        base_config['transformer_dropout_rate'] = transformer_dropout_rate
        base_config['transformer_attention_probs_dropout'] = transformer_attention_probs_dropout
        base_config['transformer_hidden_dropout_rate'] = transformer_hidden_dropout_rate
        base_config['hidden_dim_drug'] = transformer_emb_size_drug
    elif drug_encoding == 'MPNN':
        base_config['hidden_dim_drug'] = hidden_dim_drug
        base_config['batch_size'] = batch_size
        base_config['mpnn_hidden_size'] = mpnn_hidden_size
        base_config['mpnn_depth'] = mpnn_depth
    # raise NotImplementedError
    elif drug_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct drug encoding available!")

    return base_config

config = generate_config(drug_encoding = drug_encoding,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 5,
                         LR = 0.001,
                         batch_size = 128,
                         hidden_dim_drug = 700,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3
                        )
