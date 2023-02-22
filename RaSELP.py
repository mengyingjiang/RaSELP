from numpy.random import seed
import os
import gc
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from radam import RAdam
from model import *
from loss import *
from accuracy import *
from utils import *
from layers import *
import warnings
warnings.filterwarnings("ignore")


seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

bert_n_heads=4
bert_n_layers=4
cross_ver_tim=5
cov2KerSize=50
cov1KerSize=25
calssific_loss_weight=5
weight_decay_rate=0.0001


class DDIDataset(Dataset):  # [59622, 3432], [59622]
    def __init__(self, y, z):
        self.len = len(y)
        self.y_data = torch.from_numpy(y)
        self.z_data = torch.from_numpy(z)

    def __getitem__(self, index):
        return self.y_data[index], self.z_data[index]

    def __len__(self):
        return self.len

def RaSELP_train(model, y_train,  y_test_2, y_test_3, event_num, X_vector, adj, ddi_edge_train,ddi_edge_test_2,ddi_edge_test_3, multi_graph ,drug_coding):  # model [29811,3432],[29811],[7453,3432],[7453],65

    model_optimizer = RAdam(model.parameters(), lr= learn_rating, weight_decay=weight_decay_rate)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    ddi_edge_train = np.vstack((ddi_edge_train,np.vstack((ddi_edge_train[:,1],ddi_edge_train[:,0])).T))
    y_train = np.hstack((y_train, y_train))

    N_edges = ddi_edge_train.shape
    index = np.arange(N_edges[0])
    np.random.seed(seed)
    np.random.shuffle(index)

    y_train = y_train[index]
    ddi_edge_train = ddi_edge_train[index]

    len_train = len(y_train)
    len_test_2 = len(y_test_2)
    len_test_3 = len(y_test_3)
    print("arg train len", len(y_train))
    print("test len_2", len(y_test_2))
    print("test len_3", len(y_test_3))

    train_dataset = DDIDataset(ddi_edge_train, np.array(y_train))  # [59622, 3432], [59622]
    test_dataset_2 = DDIDataset(ddi_edge_test_2, np.array(y_test_2))  # [7453, 3432], [7453]
    test_dataset_3 = DDIDataset(ddi_edge_test_3, np.array(y_test_3))  # [7453, 3432], [7453]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader_2 = DataLoader(dataset=test_dataset_2, batch_size=batch_size, shuffle=False)
    test_loader_3 = DataLoader(dataset=test_dataset_3, batch_size=batch_size, shuffle=False)

    multi_graph = multi_graph.to(device)
    for epoch in range(epo_num):
        my_loss = my_loss1()
        running_loss = 0.0
        x_vector = X_vector.to(device)
        drug_coding = drug_coding.to(device)
        adj = adj.to(device)

        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
            train_edge, train_edge_labels = data
            lam = np.random.beta(0.5, 0.5)
            # lam=1
            index = torch.randperm(train_edge.size()[0]).cuda()
            targets_a, targets_b = train_edge_labels, train_edge_labels[index]
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            train_edge = torch.tensor(train_edge, dtype=torch.long)
            train_edge = train_edge.to(device)
            train_edge_mixup = train_edge[index, :]
            model_optimizer.zero_grad()

            X,X_embed_all, X_embed_all_recon, X_initial, X_initial_recon = model(multi_graph,adj,x_vector, train_edge, train_edge_mixup, lam, drug_coding)
            loss = lam * my_loss(X, targets_a,X_embed_all, X_embed_all_recon, X_initial, X_initial_recon ) \
                   + (1 - lam) * my_loss(X, targets_b,X_embed_all, X_embed_all_recon, X_initial, X_initial_recon )
            # loss = my_loss(X, targets_a,X_embed_all, X_embed_all_recon, X_initial, X_initial_recon )

            loss.backward()
            model_optimizer.step()
            running_loss += loss.item()
            del X
            del X_embed_all
            del X_embed_all_recon
            del X_initial
            del X_initial_recon
            del targets_a
            del targets_b
            del train_edge
            del loss
            gc.collect()
        # 循环完 该epoch的训练结束
        model.eval()
        testing_loss_2 = 0.0
        testing_loss_3 = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader_2, 0):
                test_edge, test_edge_labels = data
                test_edge = torch.tensor(test_edge, dtype=torch.long)
                test_edge = test_edge.to(device)
                lam = 1

                test_edge_labels = test_edge_labels.type(torch.int64).to(device)
                X, _, _, _, _ = model(multi_graph, adj, x_vector, test_edge,test_edge,lam, drug_coding)
                loss = torch.nn.functional.cross_entropy(X, test_edge_labels)
                testing_loss_2 += loss.item()

            del test_edge
            gc.collect()

            for batch_idx, data in enumerate(test_loader_3, 0):
                test_edge, test_edge_labels = data
                test_edge = torch.tensor(test_edge, dtype=torch.long)
                test_edge = test_edge.to(device)
                lam = 1
                test_edge_labels = test_edge_labels.type(torch.int64).to(device)
                X, _, _, _, _ = model(multi_graph, adj, x_vector, test_edge, test_edge, lam, drug_coding)
                loss = torch.nn.functional.cross_entropy(X, test_edge_labels)
                testing_loss_3 += loss.item()
            del test_edge
            gc.collect()


        print('epoch [%d] trn_los: %.6f tet_los_2: %.6f tet_los_3: %.6f ' % (
            epoch + 1, running_loss / len_train, testing_loss_2 / len_test_2, testing_loss_3 / len_test_3))

    pre_score_2 = np.zeros((0, event_num), dtype=float)
    pre_score_3 = np.zeros((0, event_num), dtype=float)

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader_2, 0):
            test_edge, _ = data
            test_edge = torch.tensor(test_edge, dtype=torch.long)
            test_edge = test_edge.to(device)
            lam = 1
            X, _, _, _, _= model(multi_graph, adj, x_vector, test_edge, test_edge,lam, drug_coding)
            pre_score_2 = np.vstack((pre_score_2, F.softmax(X).cpu().numpy()))

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader_3, 0):
            test_edge, _ = data
            test_edge = torch.tensor(test_edge, dtype=torch.long)
            test_edge = test_edge.to(device)
            lam = 1
            X, _, _, _, _= model(multi_graph, adj, x_vector, test_edge, test_edge,lam, drug_coding)
            pre_score_3= np.vstack((pre_score_3, F.softmax(X).cpu().numpy()))

    del model
    del X
    del model_optimizer
    del train_loader
    del test_loader_2
    del test_loader_3
    del train_dataset
    del test_dataset_2
    del test_dataset_3
    gc.collect()

    return pre_score_2,pre_score_3
