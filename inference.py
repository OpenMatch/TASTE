import os

import faiss
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from transformers import T5Tokenizer

from src.model import TASTEModel
from utils.data_loader import load_item_name, load_item_address, load_data, load_item_data, SequenceDataset, ItemDataset
from utils.option import Options
from utils.rec_metrics import get_metrics_dict
from utils.util import set_randomseed, init_logger

def evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_emb_list = []
    seq_emb_list = []
    target_item_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_item_dataloader):
            item_inputs = batch["item_ids"].to(device)
            item_masks = batch["item_masks"].to(device)
            _,item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())
        item_emb_list = np.concatenate(item_emb_list, 0)
        for i, batch in enumerate(test_seq_dataloader):
            seq_inputs = batch["seq_ids"].to(device)
            seq_masks = batch["seq_masks"].to(device)
            batch_target = batch["target_list"]
            _, seq_emb = model(seq_inputs, seq_masks)
            seq_emb_list.append(seq_emb.cpu().numpy())
            target_item_list.extend(batch_target)
        seq_emb_list = np.concatenate(seq_emb_list, 0)
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(768)
        cpu_index.add(np.array(item_emb_list, dtype=np.float32))
        query_embeds = np.array(seq_emb_list, dtype=np.float32)
        D, I = cpu_index.search(query_embeds, max(Ks))
        n_item = item_emb_list.shape[0]
        n_seq = seq_emb_list.shape[0]
        metrics_dict = get_metrics_dict(I, n_seq, n_item, Ks, target_item_list)
        logging.info(
            'Test: Recall@10：{:.4f}, Recall@20:{:.4f}, NDCG@10: {:.4f}, NDCG@20:{:.4f}'.format(
                metrics_dict[10]['recall'], metrics_dict[20]['recall'],
                metrics_dict[10]['ndcg'],metrics_dict[20]['ndcg']
            ))

        logging.info('***** Finish test *****')

def main():
    options = Options()
    opt = options.parse()
    set_randomseed(opt.seed)
    checkpoint_path = os.path.join(opt.checkpoint_dir, opt.data_name)
    checkpoint_path = os.path.join(checkpoint_path, opt.experiment_name)
    checkpoint_path = os.path.join(checkpoint_path, 'test')
    runlog_path = os.path.join(checkpoint_path, 'log')
    os.makedirs(runlog_path, exist_ok=True)
    logging = init_logger(
        os.path.join(runlog_path, 'runlog.log')
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(opt.best_model_path)
    model_class = TASTEModel
    model = model_class.from_pretrained(opt.best_model_path)
    model.to(device)

    data_dir = os.path.join(opt.data_dir, opt.data_name)
    test_file = os.path.join(data_dir, 'test.txt')
    item_file = os.path.join(data_dir, 'item.txt')

    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if opt.data_name == 'beauty' or opt.data_name == 'sports' or opt.data_name == 'toys':
        item_desc = load_item_name(item_file)
    elif opt.data_name == 'yelp':
        item_desc = load_item_address(item_file)
    item_len = len(item_desc)
    logging.info(f"item len: {item_len}")
    test_data = load_data(test_file, item_desc)
    logging.info(f"test len: {len(test_data)}")
    item_data = load_item_data(item_desc)

    test_seq_dataset = SequenceDataset(test_data, tokenizer, opt)
    test_seq_sampler = SequentialSampler(test_seq_dataset)
    test_seq_dataloader = DataLoader(
        test_seq_dataset,
        sampler=test_seq_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_seq_dataset.collect_fn
    )
    test_item_dataset = ItemDataset(item_data, tokenizer, opt)
    test_item_sampler = SequentialSampler(test_item_dataset)
    test_item_dataloader = DataLoader(
        test_item_dataset,
        sampler=test_item_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=test_item_dataset.collect_fn
    )
    Ks = eval(opt.Ks)
    evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging)






if __name__ == '__main__':
    main()