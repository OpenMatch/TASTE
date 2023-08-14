"""
We used RecBole's evaluation code:https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py.
########################
"""
import torch
import numpy as np

def recall(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
def ndcg(pos_index,pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)
    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]
    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result


def get_metrics_dict(rank_indices, n_seq, n_item, Ks,target_item_list):
    rank_indices = torch.tensor(rank_indices)
    pos_matrix = torch.zeros([n_seq,n_item], dtype=torch.int)
    for i in range(n_seq):
        # item id starts from 1
        pos = target_item_list[i] - 1
        pos_matrix[i][pos] = 1
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=rank_indices)
    pos_idx = pos_idx.to(torch.bool).cpu().numpy()
    pos_len_list = pos_len_list.squeeze(-1).cpu().numpy()
    recall_result = recall(pos_idx, pos_len_list)
    avg_recall_result = recall_result.mean(axis=0)
    ndcg_result = ndcg(pos_idx, pos_len_list)
    avg_ndcg_result = ndcg_result.mean(axis=0)
    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['recall'] = round(avg_recall_result[k - 1], 4)
        metrics_dict[k]['ndcg'] = round(avg_ndcg_result[k - 1], 4)
    return metrics_dict