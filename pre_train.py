import torch
import GCL.losses as L
import GCL.augmentors as A
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.loader import DataLoader
from brain_dataset import BrainDataset
from model import GConv,Encoder,save_encoder
import matplotlib.pyplot as plt
import argparse

def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index,data.edge_attr,data.batch)
        #g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index,data.edge_attr,data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,help='Which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=16,help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,help='Learning rate')
    parser.add_argument('--emb_dim', type=int, default=64,help='Embedding dimensions')
    parser.add_argument('--num_layer', type=int, default=4,help='Number of GNN message passing layers')
    parser.add_argument('--tau', type=float, default=0.2,help='parameters tau of InfoNCE')
    parser.add_argument('--save_path', type=str, default='saved_models',help='path for saving model')
    args = parser.parse_args()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    dataset = BrainDataset("Braindata")
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    input_dim = max(dataset.num_features, 1)

    #aug1 = A.Identity()
    aug1=A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1001, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
    gconv = GConv(input_dim=input_dim, hidden_dim=args.emb_dim, num_layers=args.num_layer).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.tau), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    loss_list=[]
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            loss_list.append(loss)

            if(epoch%1000==0):save_encoder(encoder_model.encoder,epoch,args.save_path)
    
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #test_result = test(encoder_model, dataloader)
    #print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
