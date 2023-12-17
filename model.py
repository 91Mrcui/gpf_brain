import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,GlobalAttention,GCNConv,GATConv,GINConv,GraphNorm
import os

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_attr,lengths,batch,prompt=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,edge_attr)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index,edge_attr)
        z, g = self.encoder(x, edge_index, edge_attr,lengths,batch,prompt)
        z1, g1 = self.encoder(x1, edge_index1, edge_weight1,lengths,batch)
        z2, g2 = self.encoder(x2, edge_index2, edge_weight2,lengths,batch)
        return z, g, z1, z2, g1, g2

def save_encoder(encoder_model,epoch,save_path='saved_models'):
    os.makedirs(save_path, exist_ok=True)
    torch.save(encoder_model.state_dict(), os.path.join(save_path, f'encoder_epoch_{epoch}.pth'))

def load_encoder(encoder_model, model_path):
    encoder_model.load_state_dict(torch.load(model_path))
    encoder_model.eval()
    return encoder_model

def make_gconv(input_dim,out_dim,gtype):
    if gtype == 'gcn':
        return GCNConv(in_channels=input_dim, out_channels=out_dim)
    elif gtype == 'gat':
        return GATConv(in_channels=input_dim, out_channels=out_dim, heads=4)

def make_fconv(out_dim,num_layers,ftype):
    if ftype == 'gru':
        return nn.GRU(input_size=1, hidden_size=out_dim, num_layers=num_layers, batch_first=True)
    elif ftype == 'lstm':
        return nn.LSTM(input_size=1, hidden_size=out_dim, num_layers=num_layers, batch_first=True)
    elif ftype == 'transfomer':
        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=7,batch_first=True)
        bn=nn.BatchNorm1d()
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        return transformer_encoder

class Brain_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, f_num_layers=1,g_num_layers=4,JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gcn", fnn_type='gru'):
        super(Brain_GNN, self).__init__()
        self.input_dim=input_dim
        self.hidden_ls=hidden_dim_list
        self.f_num_layers=f_num_layers
        self.g_num_layers=g_num_layers
        self.JK=JK
        self.drop_ratio=drop_ratio        


        self.f_convs=make_fconv(input_dim,f_num_layers,fnn_type)
        self.g_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.active_funs = nn.ModuleList()

        for i in range(self.g_num_layers):
            if i == 0:
                self.g_convs.append(make_gconv(input_dim, self.hidden_ls[i],gnn_type))
            else:
                self.g_convs.append(make_gconv(self.hidden_ls[i-1], self.hidden_ls[i],gnn_type))
            self.batch_norms.append(GraphNorm(self.hidden_ls[i]))
            self.active_funs.append(nn.PReLU())

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(nn.Linear(self.hidden_ls[-1], 1)))

        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_ls[-1], self.hidden_ls[-1]),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hidden_ls[-1], self.hidden_ls[-1]))

    def forward(self, x, edge_index,edge_attr,lengths,batch,prompt=None):
        #-------------1. pre deal with different lenght x-----------------------------------
        x=x.unsqueeze(-1)
        split_size = int(x.shape[0]/lengths.shape[0])
        split_x = torch.split(x, split_size, dim=0)
        outputs = []
        for i, input_tensor in enumerate(split_x):
            input_tensor = input_tensor[:, :lengths[i].item()]
            output,h_n = self.f_convs(input_tensor)
            h_n = h_n[-1:]
            x = torch.cat(h_n.split(1), dim=-1).squeeze(0)
            outputs.append(x)
        x = torch.cat(outputs, dim=0)
        
        #-------------2. prompt-------------------------------------------------------------
        if(prompt!=None):
            h=prompt.add(x)
        else:
            h = x
            
        h_list = []
        for conv, bn, af in zip(self.g_convs, self.batch_norms, self.active_funs):
            h = conv(h, edge_index,edge_attr)
            # h = bn(h)
            h = af(h)
            h = F.dropout(h, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_emb = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_emb = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        graph_emb = self.pool(node_emb, batch)
        return node_emb, graph_emb

class Task_MLP(torch.nn.Module):
    def __init__(self,encoder,emb_dim,hidden_ls,num_layer=2,task_num=2):
        super(Task_MLP, self).__init__()
        self.encoder=encoder
        self.emb_dim=emb_dim
        self.hidden_ls=hidden_ls
        self.task_num=task_num
        self.graph_pred_linear = torch.nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                #print(self.emb_dim, self.hidden_ls[i])
                self.graph_pred_linear.append(torch.nn.Linear(self.emb_dim, self.hidden_ls[i]))
            else:
                #print(self.hidden_ls[i-1],self.hidden_ls[i])
                self.graph_pred_linear.append(torch.nn.Linear(self.hidden_ls[i-1],self.hidden_ls[i]))
            self.graph_pred_linear[-1].reset_parameters()
            self.graph_pred_linear.append(torch.nn.ReLU())

        #print(self.hidden_ls[num_layer-1], self.task_num)
        if num_layer==0:
            self.graph_pred_linear.append(torch.nn.Linear(self.emb_dim, self.task_num))
        else:
            self.graph_pred_linear.append(torch.nn.Linear(self.hidden_ls[num_layer-1], self.task_num))
        self.graph_pred_linear[-1].reset_parameters()

    def from_pretrained(self, model_file):
        self.encoder.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, edge_attr,lengths,batch,prompt=None):
        node_emb, graph_emb= self.encoder(x,edge_index,edge_attr,lengths,batch,prompt)
        pred=graph_emb
        for moudel in self.graph_pred_linear:
            pred=moudel(pred)
        return pred

