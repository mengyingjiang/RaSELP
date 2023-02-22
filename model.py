from layers import *
import torch
import torch.nn.functional as F





class RaSELP(torch.nn.Module):
    def __init__(self, initial_dim, event_num,n_hop):
        super( RaSELP, self).__init__()
        input_dim = 2*graph_dim

        self.RGCN = RGCN(event_num+1, initial_dim // 2, graph_dim)
        self.GCN1 = HomoGCN(graph_dim, graph_dim,n_hop)
        self.GCN2 = HomoGCN(graph_dim, graph_dim,n_hop)
        self.GCN3 = HomoGCN(graph_dim, graph_dim,n_hop)

        self.ae1 = AE1(input_dim, dim_AE)  # Joining together
        self.ae2 = AE1(initial_dim, dim_AE)  # Joining together
        self.cnn_concat = CNN_concat(dim_AE, 'drug', **config)

        self.dr = torch.nn.Dropout(drop_out_rating)

        N_Fea = 4*dim_AE
        self.AN = torch.nn.LayerNorm(N_Fea)
        self.l1 = torch.nn.Linear(N_Fea, (N_Fea + event_num))
        self.bn1 = torch.nn.BatchNorm1d((N_Fea + event_num))
        self.l2 = torch.nn.Linear((N_Fea + event_num), event_num)
        self.ac = gelu

    def forward(self, multi_graph, label_graph, x_initial, ddi_edge, ddi_edge_mixup, lam, drug_coding):
        x_embed_known = F.relu(self.RGCN(x_initial, label_graph))
        x_embed_all = F.relu(self.dr(self.GCN1(x_embed_known, multi_graph[0, :, :])))
        x_embed_all = x_embed_all + F.relu(self.dr(self.GCN2(x_embed_known, multi_graph[1, :, :])))
        x_embed_all = x_embed_all + F.relu(self.dr(self.GCN3(x_embed_known, multi_graph[2, :, :])))

        node_id = ddi_edge.T
        node_id_mixup = ddi_edge_mixup.T
        X_smile = lam * torch.cat([drug_coding[node_id[0]], drug_coding[node_id[1]]], dim=2) \
                + (1 - lam) * torch.cat([drug_coding[node_id_mixup[0]], drug_coding[node_id_mixup[1]]], dim=2)

        x_embed_all = lam * torch.cat([x_embed_all[node_id[0]], x_embed_all[node_id[1]]], dim=1) \
                    + (1 - lam) * torch.cat([x_embed_all[node_id_mixup[0]], x_embed_all[node_id_mixup[1]]], dim=1)

        x_initial = lam * torch.cat([x_initial[node_id[0]], x_initial[node_id[1]]], dim=1) \
                  + (1 - lam) * torch.cat([x_initial[node_id_mixup[0]], x_initial[node_id_mixup[1]]], dim=1)

        X_smile = self.cnn_concat(X_smile)

        X, x_embed_all_recon = self.ae1(x_embed_all)

        X1, x_initial_recon = self.ae2(x_initial)

        X = torch.cat((X, X1, X_smile, X+ X1+X_smile), 1)
        X = self.AN(X)
        X = self.dr(self.bn1(self.ac(self.l1(X))))
        X = self.l2(X)

        return X, x_initial, x_initial_recon, x_initial, x_initial_recon