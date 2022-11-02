# import torch
# import pandas as pd
# import os
# import os.path as osp
# from datetime import date
# import shutil
# from tqdm import tqdm
# import numpy as np

# class DatasetSaver(object):
#     '''
#         A class for saving graphs and split in OGB-compatible manner
#         Create submission_datasetname/ directory, and output the following two files:
#             - datasetname.zip (OGB-compatible zipped dataset folder)
#             - meta_dict.pt (torch files storing all the necessary dataset meta-information)
#     '''
#     def __init__(self, dataset_name, is_hetero, version, root = 'submission'):
#         # verify input
#         if not ('ogbn-' in dataset_name or 'ogbl-' in dataset_name or 'ogbg-' in dataset_name):
#             raise ValueError('Dataset name must have valid ogb prefix (e.g., ogbn-*).')
#         if not isinstance(is_hetero, bool):
#             raise ValueError('is_hetero must be of type bool.')
#         if not (isinstance(version, int) and version >= 0):
#             raise ValueError('version must be of type int and non-negative')

#         self.dataset_name = dataset_name

#         self.is_hetero = is_hetero
#         self.version = version
#         self.root = root
#         self.dataset_prefix = dataset_name.split('-')[0] # specify the task category
#         self.dataset_suffix = '_'.join(dataset_name.split('-')[1:])
#         self.submission_dir = self.root + '_' + self.dataset_prefix + '_' + self.dataset_suffix
#         self.dataset_dir = osp.join(self.submission_dir, self.dataset_suffix) 
#         self.meta_dict_path = osp.join(self.submission_dir, 'meta_dict.pt')
        
#         if self.dataset_prefix == 'ogbg' and self.is_hetero:
#             raise NotImplementedError('Heterogeneous graph dataset object has not been implemented for graph property prediction yet.')

#         if osp.exists(self.dataset_dir):
#             if input(f'Found an existing submission directory at {self.submission_dir}/. \nWill you remove it? (y/N)\n').lower() == 'y':
#                 shutil.rmtree(self.submission_dir)
#                 print('Removed existing submission directory')
#             else:
#                 print('Process stopped.')
#                 exit(-1)


#         # make necessary dirs
#         self.raw_dir = osp.join(self.dataset_dir, 'raw')
#         os.makedirs(self.raw_dir, exist_ok=True)
#         os.makedirs(osp.join(self.dataset_dir, 'processed'), exist_ok=True)

#         # create release note
#         with open(osp.join(self.dataset_dir, f'RELEASE_v{version}.txt'), 'w') as fw:
#             fw.write(f'# Release note for {self.dataset_name}\n\n### v{version}: {date.today()}')

#         # check list
#         self._save_graph_list_done = False
#         self._save_split_done = False
#         self._copy_mapping_dir_done = False

#         if 'ogbl' == self.dataset_prefix:
#             self._save_target_labels_done = True # for ogbl, we do not need to give predicted labels
#         else:
#             self._save_target_labels_done = False # for ogbn and ogbg, need to give predicted labels
        
#         self._save_task_info_done = False
#         self._get_meta_dict_done = False
#         self._zip_done = False

#     def _save_graph_list_hetero(self, graph_list):
#         dict_keys = graph_list[0].keys()
#         # check necessary keys
#         if not 'edge_index_dict' in dict_keys:
#             raise RuntimeError('edge_index_dict needs to be provided in graph objects')
#         if not 'num_nodes_dict' in dict_keys:
#             raise RuntimeError('num_nodes_dict needs to be provided in graph objects')

#         print(dict_keys)

#         # Store the following files
#         # - edge_index_dict.npz (necessary)
#         #   edge_index_dict
#         # - num_nodes_dict.npz (necessary)
#         #   num_nodes_dict
#         # - num_edges_dict.npz (necessary)
#         #   num_edges_dict
#         # - node_**.npz (optional, node_feat_dict is the default node features)
#         # - edge_**.npz (optional, edge_feat_dict the default edge features)
        
#         # extract entity types
#         ent_type_list = sorted([e for e in graph_list[0]['num_nodes_dict'].keys()])

#         # saving num_nodes_dict
#         print('Saving num_nodes_dict')
#         num_nodes_dict = {}
#         for ent_type in ent_type_list:
#             num_nodes_dict[ent_type] = np.array([graph['num_nodes_dict'][ent_type] for graph in graph_list]).astype(np.int64)
#         np.savez_compressed(osp.join(self.raw_dir, 'num_nodes_dict.npz'), **num_nodes_dict)
        
#         print(num_nodes_dict)

#         # extract triplet types
#         triplet_type_list = sorted([(h, r, t) for (h, r, t) in graph_list[0]['edge_index_dict'].keys()])
#         print(triplet_type_list)

#         # saving edge_index_dict
#         print('Saving edge_index_dict')
#         num_edges_dict = {}
#         edge_index_dict = {}
#         for triplet in triplet_type_list:
#             # representing triplet (head, rel, tail) as a single string 'head___rel___tail'
#             triplet_cat = '___'.join(triplet)
#             edge_index = np.concatenate([graph['edge_index_dict'][triplet] for graph in graph_list], axis = 1).astype(np.int64)
#             if edge_index.shape[0] != 2:
#                 raise RuntimeError('edge_index must have shape (2, num_edges)')

#             num_edges = np.array([graph['edge_index_dict'][triplet].shape[1] for graph in graph_list]).astype(np.int64)
#             num_edges_dict[triplet_cat] = num_edges
#             edge_index_dict[triplet_cat] = edge_index

#         print(edge_index_dict)
#         print(num_edges_dict)

#         np.savez_compressed(osp.join(self.raw_dir, 'edge_index_dict.npz'), **edge_index_dict)
#         np.savez_compressed(osp.join(self.raw_dir, 'num_edges_dict.npz'), **num_edges_dict)

#         for key in dict_keys:
#             if key == 'edge_index_dict' or key == 'num_nodes_dict':
#                 continue 
#             if graph_list[0][key] is None:
#                 continue

#             print(f'Saving {key}')

#             feat_dict = {}

#             if 'node_' in key:
#                 # node feature dictionary
#                 for ent_type in graph_list[0][key].keys():
#                     if ent_type not in num_nodes_dict:
#                         raise RuntimeError(f'Encountered unknown entity type called {ent_type}.')
                    
#                     # check num_nodes
#                     for i in range(len(graph_list)):
#                         if len(graph_list[i][key][ent_type]) != num_nodes_dict[ent_type][i]:
#                             raise RuntimeError(f'num_nodes mistmatches with {key}[{ent_type}]')
                    
#                     # make sure saved in np.int64 or np.float32
#                     dtype = np.int64 if 'int' in str(graph_list[0][key][ent_type].dtype) else np.float32
#                     cat_feat = np.concatenate([graph[key][ent_type] for graph in graph_list], axis = 0).astype(dtype)
#                     feat_dict[ent_type] = cat_feat
                
#             elif 'edge_' in key:
#                 # edge feature dictionary
#                 for triplet in graph_list[0][key].keys():
#                     # representing triplet (head, rel, tail) as a single string 'head___rel___tail'
#                     triplet_cat = '___'.join(triplet)
#                     if triplet_cat not in num_edges_dict:
#                         raise RuntimeError(f"Encountered unknown triplet type called ({','.join(triplet)}).")

#                     # check num_edges
#                     for i in range(len(graph_list)):
#                         if len(graph_list[i][key][triplet]) != num_edges_dict[triplet_cat][i]:
#                             raise RuntimeError(f"num_edges mismatches with {key}[({','.join(triplet)})]")

#                     # make sure saved in np.int64 or np.float32
#                     dtype = np.int64 if 'int' in str(graph_list[0][key][triplet].dtype) else np.float32
#                     cat_feat = np.concatenate([graph[key][triplet] for graph in graph_list], axis = 0).astype(dtype)
#                     feat_dict[triplet_cat] = cat_feat

#             else:
#                 raise RuntimeError(f'Keys in graph object should start from either \'node_\' or \'edge_\', but \'{key}\' given.')

#             np.savez_compressed(osp.join(self.raw_dir, f'{key}.npz'), **feat_dict)

#         print('Validating...')
#         # testing
#         print('Reading saved files')
#         graph_list_read = read_binary_heterograph_raw(self.raw_dir, False)

#         print('Checking read graphs and given graphs are the same')
#         for i in tqdm(range(len(graph_list))):
#             for key0, value0 in graph_list[i].items():
#                 if value0 is not None:
#                     for key1, value1 in value0.items():
#                         if isinstance(graph_list[i][key0][key1], np.ndarray):
#                             assert(np.allclose(graph_list[i][key0][key1], graph_list_read[i][key0][key1], rtol=1e-04, atol=1e-04, equal_nan=True))
#                         else:
#                             assert(graph_list[i][key0][key1] == graph_list_read[i][key0][key1])

#         del graph_list_read