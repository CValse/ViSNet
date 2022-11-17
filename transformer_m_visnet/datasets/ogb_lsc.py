import os
import os.path as osp
import shutil
import tarfile
from multiprocessing import Pool

from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from transformer_m_visnet.datasets.utils import data2graph, preprocess_item
import pandas as pd
from rdkit import Chem

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from functools import lru_cache

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=data2graph, AddHs=False, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        self.AddHs = AddHs


        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_H_processed.pt' if self.AddHs else 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)


    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        graph_pos_list = Chem.SDMolSupplier(osp.join(self.original_root, 'pcqm4m-v2-train.sdf'), removeHs=(not self.AddHs))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']
        graph_pos_list = list(graph_pos_list) + [None] * (len(smiles_list) - len(graph_pos_list))

        print('Converting SMILES strings and SDF Mol into graphs...')
        data_list = []
        with Pool(processes=120) as pool:
            
            iter = pool.imap(data2graph, zip(smiles_list, graph_pos_list, [self.AddHs] * len(smiles_list)))

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.from_numpy(graph['position']).to(torch.float32)
                    data_list.append(data)
                    
                except:
                    continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

class GlobalPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        return preprocess_item(item)


if __name__ == '__main__':
    dataset = GlobalPygPCQM4Mv2Dataset(root='data')
    data = dataset[0]
    print(data)
    print('edge_index', data.edge_index)
    print('edge_attr', data.edge_attr)
    print('x', data.x)
    print('pos', data.pos)
    print('attn_bias', data.attn_bias)
    print('attn_edge_type', data.attn_edge_type)
    print('spatial_pos', data.spatial_pos)
    print('in_degree', data.in_degree)
    print('out_degree', data.out_degree)
    print('edge_input', data.edge_input)