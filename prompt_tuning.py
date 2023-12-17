import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from brain_dataset import BrainDataset,get_dataset
from model import Encoder,Brain_GNN,Task_MLP,load_encoder
from gpf import GraphPrompt,GPFplusAtt,GPFwithcluster
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import random_split


def train(model,device,dataloader,optimizer,criterion,prompt):
    model.train()
    epoch_loss = 0
    for step, data in enumerate(tqdm(dataloader, desc="Train Iteration")):
        data = data.to(device)
        optimizer.zero_grad()
        pred=model(data.x, data.edge_index,data.edge_attr,data.lengths,data.batch,prompt)
        loss = criterion(pred, data.y.long())
        epoch_loss+=loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss

def eval(model, device, loader, prompt, flag=False):
    model.eval()
    y_true = []
    y_pred = []
    for step, data in enumerate(tqdm(loader, desc="Eval Iteration")):
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index,data.edge_attr,data.lengths,data.batch,prompt)
            pred = out.argmax(dim=1)

            y_true.append(data.y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    if flag:
        print(y_true)
        print(y_pred)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return accuracy




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,help='Which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=8,help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,help='Learning rate')
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.2,help='parameters tau of InfoNCE')
    parser.add_argument('--prompt_type',type=str,default='gpf-plus',help='type of graph prompt feature')
    parser.add_argument('--model_path', type=str, default='saved_models/encoder_epoch_86.pth' ,help='path of pretrained model')
    parser.add_argument('--task_num',type=int,default=5,help='num of output')
    
    args = parser.parse_args()
    
    device = torch.device('cuda')
    #path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = get_dataset('prompt')
    all_size = len(dataset)
    # 6:2:2
    train_size = int(0.6 * all_size)
    eval_size = int(0.5*(all_size - train_size))
    test_size = all_size-train_size-eval_size

    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(4432))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    emb_dim=128
    f_num_layers=1
    g_num_layers=4
    JK='last'
    drop_ratio=0.
    graph_pool='mean'
    gnn_type='gcn'
    fnn_type='gru'
    hidden_dim_list=[emb_dim for i in range(g_num_layers)]
    
    gconv = Brain_GNN(input_dim=emb_dim, hidden_dim_list=hidden_dim_list, f_num_layers=f_num_layers,
                      g_num_layers=g_num_layers,JK=JK, drop_ratio=drop_ratio, graph_pooling=graph_pool, 
                      gnn_type=gnn_type, fnn_type=fnn_type).to(device)

    pretrained_model_path = args.model_path 
    if os.path.exists(pretrained_model_path):
        print(f'Loading pre-trained model from {pretrained_model_path}')
        gconv=load_encoder(gconv, pretrained_model_path)

    hidden_ls=[64,32]
    num_layer=len(hidden_ls)
    model = Task_MLP(encoder=gconv,emb_dim=emb_dim,hidden_ls=hidden_ls,num_layer=num_layer,task_num=args.task_num).to(device)

    if args.prompt_type=='gpf':
        prompt = GraphPrompt(emb_dim).to(device)    
    elif args.prompt_type=='gpf-plus':
        prompt = GPFplusAtt(emb_dim,p_num=8).to(device) 
    elif args.prompt_type=='gpf-cluster':
        prompt = GPFwithcluster(emb_dim).to(device)
    else:
        raise ValueError("Invalid prompt type. Use 'gpf', 'gpf-plus', or 'gpf-cluster'") 
    model_param_group = []
    model_param_group.append({"params": prompt.parameters(), "lr": args.lr})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr})
    optimizer = torch.optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    criterion = nn.CrossEntropyLoss()

    train_lost_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.epochs):
        loss=train(model,device,train_loader,optimizer,criterion,prompt)
        print(loss)
        train_lost_list.append(loss)
        acc = eval(model, device, eval_loader, prompt,True)
        print(acc)
        val_acc_list.append(acc)
    test_acc = eval(model, device, test_loader, prompt, True)

if __name__ == '__main__':
    main()