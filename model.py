import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool,GCNConv
import os

def make_gcn_conv(input_dim,out_dim):
    return GCNConv(in_channels=input_dim, out_channels=out_dim)

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gcn_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gcn_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        # proj head
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

        self.projection_head = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index,edge_attr,batch,prompt=None):
        if(prompt!=None):
            z=prompt.add(x)
        else:
            z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            #print(z.dtype,z.shape,edge_index.dtype)
            z = conv(z, edge_index,edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)

        # proj
        z=self.projection_head(z)
        zs.append(z)

        # 对每一层的输出进行全局池化，得到节点级别和图级别的特征
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_attr,batch,prompt=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index,edge_attr)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index,edge_attr)
        z, g = self.encoder(x, edge_index, edge_attr,batch,prompt)
        z1, g1 = self.encoder(x1, edge_index1, edge_weight1,batch)
        z2, g2 = self.encoder(x2, edge_index2, edge_weight2,batch)
        return z, g, z1, z2, g1, g2

class Downstream(torch.nn.Module):
    def __init__(self,encoder,emb_dim,num_layer=2,task_num=2):
        self.encoder=encoder
        self.emb_dim=emb_dim
        self.task_num=task_num
        self.graph_pred_linear = torch.nn.ModuleList()

        #self.head_proj=torch.nn.Linear()

        for i in range(num_layer - 1):
            self.graph_pred_linear.append(torch.nn.Linear(self.emb_dim, self.emb_dim))
            self.graph_pred_linear[-1].reset_parameters()
            self.graph_pred_linear.append(torch.nn.ReLU())
        self.graph_pred_linear.append(torch.nn.Linear(self.emb_dim, self.task_num))
        self.graph_pred_linear[-1].reset_parameters()

    def from_pretrained(self, model_file):
        # self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, edge_attr,batch,prompt=None):
        z, g= self.encoder(x,edge_index,edge_attr,batch,prompt)
        pred=self.graph_pred_linear(pred)
        return pred

        

def save_encoder(encoder_model,epoch,save_path='saved_models'):
    os.makedirs(save_path, exist_ok=True)
    torch.save(encoder_model.state_dict(), os.path.join(save_path, f'encoder_epoch_{epoch}.pth'))

def load_encoder(encoder_model, model_path):
    encoder_model.load_state_dict(torch.load(model_path))
    encoder_model.eval()
    return encoder_model