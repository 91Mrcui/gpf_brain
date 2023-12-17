import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot
from sklearn.cluster import KMeans

class GraphPrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(GraphPrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb



class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p

# Setting prompts based on clustering
class GPFwithcluster(nn.Module):
    def __init__(self, in_channels: int, k: int):
        super(GPFwithcluster, self).__init__()
        self.kmeans = KMeans(n_clusters=k, random_state=0)
        self.prompts = nn.ParameterList([nn.Parameter(torch.Tensor(1, in_channels)) for _ in range(k)])        
        glorot(self.prompts)

    def forward(self, x: Tensor):
        # Cluster the input feature matrix x
        cluster_labels = self.kmeans.fit_predict(x.cpu().detach().numpy())
        # Apply the corresponding prompts to each clustering
        cluster_prompts = [self.prompts[label] for label in cluster_labels]
        cluster_prompts = torch.cat(cluster_prompts, dim=0)

        x = x + cluster_prompts.to(x.device)
        return x

# GPF with pool
class GPF_Pool(nn.Module):
    def __init__(self, emb_dim:int, k:int, N:int):
        super(GPF_Pool,self).__init__()
        self.K=k
        self.prompts = nn.ParameterList([nn.Parameter(torch.Tensor(1,emb_dim)) for _ in range(N)])
        self.keys = nn.ParameterList([nn.Parameter(torch.Tensor(1,emb_dim)) for _ in range(N)])
        glorot(self.prompts)

    def forward(self, x: Tensor, query: Tensor):
        similarities = F.cosine_similarity(query.unsqueeze(0), torch.cat(self.keys), dim=1)
        topk_indices = torch.topk(similarities, self.K).indices
        selected_prompts = [self.prompts[i] for i in topk_indices]
        selected_prompts = torch.cat(selected_prompts, dim=0)
        x = x + selected_prompts.to(x.device)
        return x



   