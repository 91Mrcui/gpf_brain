import torch
from torch_geometric.data import Data,InMemoryDataset,Dataset
from torch_geometric.loader import DataLoader
import os
import math
import numpy as np
import os.path as osp
import pandas as pd
import re
from torch.utils.data import random_split
# Number of samples
NUM_GRAPH=392
# Number of adj matrix
NUM_FEQ=5

MAX_LEN=193


class BrainDataset(Dataset):
    def __init__(self, root, filepath="data",transform=None, pre_transform=None):
        self.filepath=filepath
        self.filenames = os.listdir(filepath)
        super(BrainDataset,self).__init__(root, transform, pre_transform)
        #self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        #names=['']
        #return ['some_file_1', 'some_file_2', ...]
        return self.filenames
    @property
    def processed_file_names(self):
        return [f'brain_data_{i}.pt' for i in range(1,NUM_GRAPH*5+1)]
        #return ['brain_data_1.pt','brain_data_2.pt','brain_data_3.pt','brain_data_4.pt','brain_data_5.pt']

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'brain_data_{}.pt'.format(idx+1)))

        return data


    def build_label(self,file_path,idx):
        file=file_path+'\慢性主观性耳鸣数据.csv'
        df = pd.read_csv('chronic_tinnitus_data\慢性主观性耳鸣数据.csv',encoding="gbk")
        id_list = df['编号'].tolist()
        y_list = df['耳鸣分级'].tolist()

        id_list = [x for x in id_list if pd.notna(x)]
        y_list = [x for x in y_list if pd.notna(x)]

        for i in range(len(id_list)):
            matches = re.search(r'\d{3}$', id_list[i])
            id = int(matches.group())
            if id == idx:
                print(idx,y_list[i])
                return y_list[i]


    def build_node_feature(self,file_path,p_id,pos=False):
        X=[]
        if p_id < 10:
            files = os.listdir(f'{file_path}01_01_sub00{p_id}')
        elif p_id < 100:
            files = os.listdir(f'{file_path}01_01_sub0{p_id}')
        else:
            files = os.listdir(f'{file_path}01_01_sub{p_id}')
           # 读入文件夹
        num_seg = len(files)
        for s_id in range(1,num_seg+1):
            if p_id < 10:   
                curr_path=f"{file_path}01_01_sub00{p_id}\\01_01_sub00{p_id}_{s_id}-SqrAct@Rois.txt"
            elif p_id < 100:
                curr_path=f"{file_path}01_01_sub0{p_id}\\01_01_sub0{p_id}_{s_id}-SqrAct@Rois.txt"
            else:
                curr_path=f"{file_path}01_01_sub{p_id}\\01_01_sub{p_id}_{s_id}-SqrAct@Rois.txt"
            with open(curr_path,"r") as file:
                lines=file.readlines()
            data = [[float(value) for value in line.split()] for line in lines]
            numpy_array = np.array(data)
            column_means = np.mean(numpy_array, axis=0).tolist()
            #print(len(column_means))
            X.append(column_means)
        numpy_array = np.array(X)
        if(pos==True):
            with open(file_path+"ROI220_nearst-ROIcentroids.txt","r") as pos_file:
                ls=pos_file.readlines()
            pos_data = [list(map(float, l.split()[:-1])) for l in ls[1:]]
            numpy_array2 = np.array(pos_data)
            numpy_array=np.vstack((numpy_array,numpy_array2.T))
        #print(numpy_array.shape)
        torch_tensor = torch.from_numpy(numpy_array).T.to(torch.float32)
        return torch_tensor

    def build_edge(self,file_path,p_index,feq_index):
        if p_index < 10:   
            file_path=f'{file_path}01_01_sub00{p_index}-ROIlaggedCoh.txt'
        elif p_index < 100:
            file_path=f'{file_path}01_01_sub0{p_index}-ROIlaggedCoh.txt'
        else:
            file_path=f'{file_path}01_01_sub{p_index}-ROIlaggedCoh.txt'
        with open(file_path, 'r') as file:
            for _ in range(14):
                next(file)
            content = file.read()
        matrices_text = content.split('\n\n')
        adjacency_matrices = []
        for matrix_text in matrices_text:
            if not matrix_text.strip():
                continue
            matrix_lines = matrix_text.splitlines()
            matrix = np.loadtxt(matrix_lines)        
            adjacency_matrices.append(matrix)
        adjacency_matrices_tensor = torch.tensor(adjacency_matrices[feq_index], dtype=torch.float32)
        
        '''
        This function is used to reduce the edge
        pos_data : Whether to consider location information
        threshold_type : mean/median/topk 
        percentile : For topk , if you want 20%, enter 20
        scale_factor: For adjusting the effect of position information 
        '''
        #adjacency_matrices_tensor=self.threshold_adjacency_matrix(adjacency_matrix=adjacency_matrices_tensor,pos_data=False,threshold_type='mean',percentile=20,scale_factor=0.01)
        
        
        nonzero_indices = torch.nonzero(adjacency_matrices_tensor, as_tuple=False)
        #nonzero_indices = self.k_nearest_neighbors(K=4)

        edge_weights = adjacency_matrices_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1]].clone().detach()

        #with open("record.txt",'w') as fp:
        #    for i in range(len(nonzero_indices)):
        #        fp.write(str(nonzero_indices[i])+'  '+str(edge_weights[i]))
        #        fp.write('\n')
        edge_index = nonzero_indices.t()
        
        edge_attr = edge_weights.view(-1, 1)  # 
        return edge_index,edge_attr
    
    def dis(self,pos, i, j):
        if i<0 or i>=len(pos) or j<0 or j>=len(pos):
            raise ValueError("Invalid indices")
        distance = math.sqrt((pos[i][0] - pos[j][0])**2 
                        +(pos[i][1] - pos[j][1])**2 
                        +(pos[i][2] - pos[j][2])**2)
        return distance
    
    def distance(self,a, b):
        distance = math.sqrt((a[0] - b[0])**2 
                       +(a[1] - b[1])**2 
                       +(a[2] - b[2])**2)
        return distance
    
    def k_nearest_neighbors(self,K=4):
        fp=self.filepath+"\\脑区位置信息\\ROI220_nearst-ROIcentroids.txt"
        with open(fp,"r") as pos_file:
            ls=pos_file.readlines()
            pos_data = [list(map(float, l.split()[:-1])) for l in ls[1:]]
        edge_index = []
        for i in range(len(pos_data)):
            dis_list=[]
            idx=[]
            for j in range(i+1,len(pos_data)):
                if i==j:continue
                distance=self.distance(pos_data[i],pos_data[j])
                dis_list.append(distance)
                idx.append((i,j))
            sorted_indices = sorted(range(len(dis_list)), key=lambda i: dis_list[i])
            for num in range(K):
                if(idx[sorted_indices[num]] not in edge_index):
                    src=idx[num][0]
                    dst=idx[num][1]
                    edge_index.append([src,dst])
                    edge_index.append([dst,src])
        edge_index=torch.from_numpy(np.array(edge_index)).to(torch.long)                
        return edge_index


    def threshold_adjacency_matrix(self,adjacency_matrix=None, pos_data=False,threshold_type='mean',percentile='20',scale_factor=0.01):
        if threshold_type not in ['mean', 'median', 'topk']:
            raise ValueError("Invalid threshold type. Use 'mean', 'median', or 'topk'.")
   
        if (pos_data == True):
            fp=self.filepath+"\\脑区位置信息\\ROI220_nearst-ROIcentroids.txt"
            with open(fp,"r") as pos_file:
                ls=pos_file.readlines()
                pos_data = [list(map(float, l.split()[:-1])) for l in ls[1:]]
            for i in range(adjacency_matrix.size(0)-1):
                for j in range(i+1,adjacency_matrix.size(0)):
                    adjacency_matrix[i,j]=adjacency_matrix[i,j]/(self.dis(pos_data,i,j)*scale_factor)
                    adjacency_matrix[j,i]=adjacency_matrix[i,j]

        if threshold_type == 'mean':
            threshold_value = torch.mean(adjacency_matrix)
        elif threshold_type == 'median':
            threshold_value = torch.median(adjacency_matrix)
        elif threshold_type == 'topk':
            if percentile is None or not 0 < percentile <= 100:
                raise ValueError("Invalid percentile value. Provide a valid value between 0 and 100 for 'topk' threshold type.")
            k = int((percentile / 100) * adjacency_matrix.numel())
            values, _ = torch.topk(adjacency_matrix.view(-1), k=k, largest=True)
            threshold_value = values[-1]

        # Setting elements less than the threshold to zero
        thresholded_matrix = torch.where(adjacency_matrix > threshold_value, adjacency_matrix, torch.tensor(0.0))
        return thresholded_matrix


    def process(self):
        data_list=[]
        cnt=0
        for i in range(1,421):
            if i < 10:
                filesp = f'{self.filepath}\\脑区功率值\\01_01_sub00{i}'
            elif i < 100:
                filesp = f'{self.filepath}\\脑区功率值\\01_01_sub0{i}'
            else:
                filesp = f'{self.filepath}\\脑区功率值\\01_01_sub{i}'
            if not os.path.exists(filesp):
                continue
            cnt+=1
            label=self.build_label(self.filepath,i)
            for j in range(NUM_FEQ):
                #print(filesp,j)
                X=self.build_node_feature(self.filepath+'\\脑区功率值\\',i,False)
                padded_X = torch.zeros((X.shape[0], MAX_LEN))
                padded_X[:, :X.shape[1]] = X
                # 记录原始长度
                lengths = torch.tensor(X.shape[1])
                Edge_index,Edge_attr=self.build_edge(self.filepath+'\\脑区相干性\\',i,j)
                data=Data(x=padded_X,edge_index=Edge_index,edge_attr=Edge_attr,y=torch.tensor(label-1),lengths=lengths)
                print(data)
                torch.save(data, osp.join(self.processed_dir, 'brain_data_{}.pt'.format(NUM_FEQ*(cnt-1)+j+1)))
                #data_list.append(data)
       
        '''
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        '''

def get_dataset(set_type):
    b = BrainDataset("Braindata",filepath="chronic_tinnitus_data")
    dataset_size = len(b)
    downstream_size = int(0.4 * dataset_size)
    pre_train_size = dataset_size - downstream_size
    pre_train_dataset, test_dataset = random_split(b, [pre_train_size, downstream_size], generator=torch.Generator().manual_seed(8848))
    if set_type=='pre-train':
        return pre_train_dataset
    else :
        return test_dataset
'''
b = BrainDataset("Braindata",filepath="chronic_tinnitus_data")
for i in range(len(b)):
    print(b[i].y)
'''