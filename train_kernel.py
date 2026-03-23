import json
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from gklearn.kernels import WeisfeilerLehman
from sklearn.svm import SVC
from itertools import combinations
from sklearn.model_selection import ParameterGrid


BRAAK_TO_GROUP = {2: 0, 3: 0, 4: 1, 5: 2, 6: 2}


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

    data_path = Path('data/braak_xy_expression_by_celltype_donor_lognorm')
    
    grid = {
        'n_neighbors': [1, 5, 10, 15, 20, 25, 30],
        'height': [1, 5, 10, 15, 20],
    }
    
    df = pd.DataFrame(columns=['n_neighbors', 'height', 'avg. accuracy'])
    
    for i, gr in enumerate(ParameterGrid(grid)):
        
        print(gr)
        df.at[i, 'n_neighbors'] = gr['n_neighbors']
        df.at[i, 'height'] = gr['height']
        graphs, braaks, donor_list = get_graphs_and_braaks(data_path, gr['n_neighbors'])
        # Fitting kernel
        wl = WeisfeilerLehman(
            height=gr['height'], 
            base_kernel='subtree',
            normalize=True
        )
        mat, _ = wl.compute(graphs, verbose=True)
        
        train_idxs, test_idxs = custom_train_test_split(mat, 5)
        
        data_split = []
        acc = []
        for tr, ts in zip(train_idxs, test_idxs):
            tr = list(tr)
            ts = list(ts)
            svm = SVC(kernel='precomputed', random_state=2026)
            svm.fit(
                mat[tr, :][:, tr], 
                [braaks[idx] for idx in tr]
            )
            acc.append(
                svm.score(
                    mat[ts, :][:, tr], 
                    [braaks[idx] for idx in ts]
                )
            )
            data_split.append(
                {
                    'train': [donor_list[idx] for idx in tr],
                    'test': [donor_list[idx] for idx in ts]
                }
            )
        with open(f'folds.json', 'w') as f:
            json.dump(
                data_split,
                f,
                indent=4
            )
        df.at[i, 'avg. accuracy'] = np.mean(acc)
        
    df.to_csv('wl-kernel_cv_accuracy.csv')
            