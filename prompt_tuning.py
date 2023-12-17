import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from brain_dataset import BrainDataset
from model import Encoder,GConv,Downstream,load_encoder
from gpf import GraphPrompt,GPFplusAtt,GPFwithcluster
import os
import tqdm

def load_encoder(encoder_model, model_path):
    encoder_model.load_state_dict(torch.load(model_path))
    encoder_model.eval()
    return encoder_model

def train(model,dataloader,optimizer,criterion,prompt):
    model.train()
    for step, data in enumerate(tqdm(dataloader, desc="Iteration")):
        data = data.to('cuda')
        optimizer.zero_grad()
        
        pred=model(data.x, data.edge_index,data.edge_attr,data.batch,prompt)
        
        loss = criterion(pred.double(), data.y)
        loss.backward()

        optimizer.step()
    pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,help='Which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=16,help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=64,help='Embedding dimensions')
    parser.add_argument('--num_layer', type=int, default=4,help='Number of GNN message passing layers')
    parser.add_argument('--tau', type=float, default=0.2,help='parameters tau of InfoNCE')
    parser.add_argument('--prompt_type',type=str,default='gpf',help='type of graph prompt feature')
    parser.add_argument('--model_path', type=str, default='saved_models/encoder_epoch_1000.pth' ,help='path of pretrained model')
    parser.add_argument('--task_num',type=int,default=2,help='num of output')
    
    args = parser.parse_args()
    
    
    
    device = torch.device('cuda')
    #path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = BrainDataset("Braindata")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    input_dim = max(dataset.num_features, 1)

    
    gconv = GConv(input_dim=input_dim, hidden_dim=args.emb_dim, num_layers=args.num_layer).to(device)
    pretrained_model_path = args.model_path 
    if os.path.exists(pretrained_model_path):
        print(f'Loading pre-trained model from {pretrained_model_path}')
        gconv=load_encoder(gconv, pretrained_model_path)

    model = Downstream(encoder=gconv,emb_dim=gconv.hidden_dim,num_layer=gconv.num_layers,task_num=args.task_num).to(device)

    if args.prompt_type=='gpf':
        prompt = GraphPrompt(input_dim)    
    elif args.prompt_type=='gpf-plus':
        prompt = GPFplusAtt(input_dim) 
    elif args.prompt_type=='gpf-cluster':
        prompt = GPFwithcluster(input_dim)
    else:
        raise ValueError("Invalid prompt type. Use 'gpf', 'gpf-plus', or 'gpf-cluster'") 
    model_param_group = []
    model_param_group.append({"params": prompt.parameters()})
    model_param_group.append({"params": model.graph_pred_linear.parameters()})

    optimizer = torch.optim.Adam(model_param_group, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.epochs):
        train(model,dataloader,optimizer,criterion,prompt)


if __name__ == '__main__':
    main()