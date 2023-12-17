import torch
from torch_geometric.data import Data,InMemoryDataset
import os
import math
import numpy as np

# Number of samples
NUM_GRAPH=8
# Number of segment
NUM_SEG=53
# Number of adj matrix
NUM_FEQ=5

class BrainDataset(InMemoryDataset):
    def __init__(self, root, filepath="data",transform=None, pre_transform=None):
        self.filepath=filepath
        self.filenames = os.listdir(filepath)
        super(BrainDataset,self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        #names=['']
        #return ['some_file_1', 'some_file_2', ...]
        return self.filenames
    @property
    def processed_file_names(self):
        return ['data.pt']


    def build_node_feature(self,file_path,p_id,pos=False):
        X=[]
        for s_id in range(1,NUM_SEG+1):
            curr_path=f"{file_path}01_sub00{p_id}\\01_sub00{p_id}_{s_id}-SqrAct@Rois.txt"
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
        file_path=f'{file_path}01_sub00{p_index}.set-ROIlaggedCoh.txt'
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
        adjacency_matrices_tensor=self.threshold_adjacency_matrix(adjacency_matrix=adjacency_matrices_tensor,
                                                     pos_data=False,
                                                     threshold_type='mean',
                                                     percentile=20,
                                                     scale_factor=0.01)
                                                     
        nonzero_indices = torch.nonzero(adjacency_matrices_tensor, as_tuple=False)
        edge_weights = adjacency_matrices_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1]].clone().detach()

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
        for i in range(1,NUM_GRAPH+1):
            for j in range(NUM_FEQ):
                X=self.build_node_feature(self.filepath+'\\脑区功率值\\',i,True)
                Edge_index,Edge_attr=self.build_edge(self.filepath+'\\脑区相干性\\',i,j)
                data=Data(x=X,edge_index=Edge_index,edge_attr=Edge_attr)
                data_list.append(data)
       
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#test
b = BrainDataset("Braindata")
print(b._data.num_nodes,b._data.num_edges,b._data.num_features)