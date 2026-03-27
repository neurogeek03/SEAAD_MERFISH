import warnings
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path
from gklearn.kernels import WeisfeilerLehman
from sklearn.svm import SVC
from itertools import combinations
from sklearn.model_selection import ParameterGrid
from MKLpy.algorithms import GRAM
from MKLpy.preprocessing import kernel_centering
from MKLpy.model_selection import cross_val_score
from MKLpy.scheduler  import ReduceOnWorsening
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

BRAAK_TO_GROUP = {2: 0, 3: 0, 4: 0, 5: 1, 6: 1}


def get_graphs_and_braaks(data_path: Path, n_neighbors: int):
    graphs = []
    braaks = []
    dfs = []
    for file in data_path.rglob('*.csv'):
        dfs.append(
            pd.read_csv(file)
        )
    
    
    all_dfs = pd.concat(dfs, axis=0, ignore_index=True)
    all_dfs = all_dfs[all_dfs['BRAAK_score'] != 0]
    donor_list = all_dfs['Donor_ID'].unique().tolist()
    
    for subject in donor_list:
        
        sub_df = all_dfs[all_dfs['Donor_ID'] == subject]
        if len(sub_df['Section'].unique()) > 1:
            sub_df = sub_df[sub_df['Section'] == sub_df['Section'].unique()[0]]
            
        braak_raw = int(sub_df['BRAAK_score'].iloc[0])
        binned_score = BRAAK_TO_GROUP[braak_raw]

        braaks.append(binned_score)
        graph = kneighbors_graph(sub_df[['x','y']].to_numpy(), n_neighbors=n_neighbors, mode='connectivity', include_self=False)
        graph = nx.from_scipy_sparse_array(graph)
        labels = {x[0]: x[1] for x in enumerate(sub_df['Subclass'].to_list())}
        nx.set_node_attributes(graph, labels, 'cell_type')
        graphs.append(graph)
    
    return graphs, braaks, donor_list


def custom_train_test_split(x, folds=5):
    
    num_samples = x.shape[0]
    possible_idxs = set(range(x.shape[0]))
    
    rng = np.random.default_rng(2026)
    shuffled_idx = rng.permuted(np.arange(num_samples))
    split_dataset = np.array_split(shuffled_idx, folds)
    train_sets = list(combinations(split_dataset, folds - 1))
    train_sets = [set(np.concat(tr_set).tolist()) for tr_set in train_sets]
    test_sets = [possible_idxs - tr_set for tr_set in train_sets]
    
    return train_sets, test_sets
    
if __name__ == '__main__':
    warnings.simplefilter('ignore', UserWarning)


    data_path = Path('data/braak_xy_expression_by_celltype_donor_lognorm')
    
    params = {
        'n_neighbors': [30, 15, 70],
        'max_iter': [1000, 1500],
        'lr': [0.01, 0.1, 0.001]
    }
    
    demog = pd.read_csv('dummy_demog.csv', index_col=0)
    df = pd.DataFrame(
        columns=[
            'n_neighbors', 
            'max_iter', 
            'lr',
            'mat_rbf_acc',
            'rbf_acc',
            'mat_acc'
        ]
    )
    
    rbf = rbf_kernel(demog.to_numpy())
    rbf = kernel_centering(rbf)
    
    for i, grid in tqdm(enumerate(ParameterGrid(params)), total=len(ParameterGrid(params))):
        print(grid)
        df.at[i, 'n_neighbors'] = grid['n_neighbors']
        df.at[i, 'max_iter'] = grid['max_iter']
        df.at[i, 'lr'] = grid['lr']
        graphs, braaks, donor_list = get_graphs_and_braaks(data_path, grid['n_neighbors'])
        braaks = np.array(braaks)
        # Fitting kernel
        wl = WeisfeilerLehman(
            height=5, 
            base_kernel='subtree',
            normalize=False
        )
        mat, _ = wl.compute(graphs, verbose=True)
        mat = kernel_centering(mat)
    
        mkl = GRAM(
            max_iter=grid['max_iter'],          
            learning_rate=grid['lr'],      
            scheduler=ReduceOnWorsening()
        )
        try:
            scores = cross_val_score(
                [mat, rbf], 
                braaks, 
                mkl, 
                LeaveOneOut(), 
                random_state=42, 
                scoring='accuracy'
            )
            df.at[i, 'mat_rbf_acc'] = np.mean(scores)
            print(f'mat and rbf: {np.mean(scores):.4f}')
        
            scores = []
            for train, test in LeaveOneOut().split(braaks):
                svc = SVC(kernel='precomputed')
                svc.fit(rbf[train][:, train], braaks[train])   
                scores.append(svc.score(rbf[test][:, train], braaks[test]))
            df.at[i, 'rbf_acc'] = np.mean(scores)
            print(f'rbf {np.mean(scores)}')
        
            scores = []
            for train, test in LeaveOneOut().split(braaks):
                svc = SVC(kernel='precomputed')
                svc.fit(mat[train][:, train], braaks[train])   
                scores.append(svc.score(mat[test][:, train], braaks[test]))
            df.at[i, 'mat_acc'] = np.mean(scores)
            print(f'mat only {np.mean(scores)}')
        except Exception:
            print(f'{grid} Failed with {Exception}')
            continue

    df.to_csv('hparam_search.csv')
            
    
    
    