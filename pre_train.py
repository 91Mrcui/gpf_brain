import torch
import GCL.losses as L
import GCL.augmentors as A
#from utils import InfoNCE,EdgeRemoving,NodeDropping,DualBranchContrast
from tqdm import tqdm
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from brain_dataset import get_dataset
from model import Brain_GNN,Encoder,save_encoder
import argparse
import random

def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index,data.edge_attr,data.lengths,data.batch)
        g1, g2 = [encoder_model.encoder.projection_head(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,help='Which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=4,help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=128,help='Embedding dimensions')
    parser.add_argument('--f_num_layers', type=int, default=1,help='Number of FNN layers')
    parser.add_argument('--g_num_layers', type=int, default=4,help='Number of GNN message passing layers')
    parser.add_argument('--tau', type=float, default=0.2,help='parameters tau of InfoNCE')
    parser.add_argument('--save_path', type=str, default='saved_models',help='path for saving model')
    parser.add_argument('--JK', type=str, default='last',help='JK')
    parser.add_argument('--drop_ratio', type=float, default=0.5)
    parser.add_argument('--graph_pool', type=str ,default='mean')
    parser.add_argument('--gnn_type', type=str ,default='gcn')
    parser.add_argument('--fnn_type', type=str ,default='gru')

    args = parser.parse_args()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = get_dataset('pre-train')
    #dataset=BrainDataset("Braindata",filepath="chronic_tinnitus_data")
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    #input_dim = max(dataset.num_features, 1)

    '''
    aug1 = A.Identity()
    aug1=A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    '''
    aug_ratio = random.randint(1, 3) * 1.0 / 10
    aug1 = A.NodeDropping(pn=aug_ratio)
    aug2 = A.EdgeRemoving(pe=aug_ratio)
    

    hidden_dim_list=[args.emb_dim for i in range(args.g_num_layers)]
    gconv = Brain_GNN(input_dim=args.emb_dim, hidden_dim_list=hidden_dim_list, f_num_layers=args.f_num_layers,
                      g_num_layers=args.g_num_layers,JK=args.JK, drop_ratio=args.drop_ratio, graph_pooling=args.graph_pool, 
                      gnn_type=args.gnn_type, fnn_type=args.fnn_type).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.tau), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    loss_list=[]
    train_loss_min = 1000000
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            loss_list.append(loss)
            if train_loss_min > loss:
                train_loss_min = loss
                save_encoder(encoder_model.encoder,epoch,args.save_path)
            # if(epoch%10==0):save_encoder(encoder_model.encoder,epoch,args.save_path)
    with open('losses.txt','w')as f:
        for i in range(len(loss_list)):
            f.write(f'Epoch {i+1}, Loss {loss_list[i]}\n')
    '''
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    '''


if __name__ == '__main__':
    main()
