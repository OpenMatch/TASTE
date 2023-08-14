import glob
import os
import shutil

import faiss
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer

from src.model import TASTEModel
from utils.data_loader import load_item_name, load_item_address, load_data, load_item_data, SequenceDataset, ItemDataset
from utils.option import Options
from utils.rec_metrics import get_metrics_dict
from utils.util import set_randomseed, init_logger, early_stopping


def evaluate(model, eval_seq_dataloader, eval_item_dataloader, device, Ks, logging,tb_logger,step):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_emb_list = []
    seq_emb_list = []
    target_item_list = []
    with torch.no_grad():
        for i, batch in enumerate(eval_item_dataloader):
            item_inputs = batch["item_ids"].to(device)
            item_masks = batch["item_masks"].to(device)
            _,item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())
        item_emb_list = np.concatenate(item_emb_list, 0)
        for i, batch in enumerate(eval_seq_dataloader):
            seq_inputs = batch["seq_ids"].to(device)
            seq_masks = batch["seq_masks"].to(device)
            batch_target = batch["target_list"]
            _,seq_emb = model(seq_inputs, seq_masks)
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
        'current:step:{} Recall@10：{:.4f}, Recall@20:{:.4f}, NDCG@10: {:.4f}, NDCG@20:{:.4f}'.format(
            step,metrics_dict[10]['recall'], metrics_dict[20]['recall'],
            metrics_dict[10]['ndcg'],
            metrics_dict[20]['ndcg']))
    if tb_logger is not None:
        tb_logger.add_scalar("recall@10", metrics_dict[10]['recall'], step)
        tb_logger.add_scalar("recall@20", metrics_dict[20]['recall'], step)
        tb_logger.add_scalar("ndcg@10", metrics_dict[10]['ndcg'], step)
        tb_logger.add_scalar("ndcg@20", metrics_dict[20]['ndcg'], step)
    return metrics_dict

def main():
    options = Options()
    opt = options.parse()
    set_randomseed(opt.seed)
    checkpoint_path = os.path.join(opt.checkpoint_dir, opt.data_name)
    checkpoint_path = os.path.join(checkpoint_path, opt.experiment_name)
    checkpoint_path = os.path.join(checkpoint_path, 'eval')
    runlog_path = os.path.join(checkpoint_path, 'log')
    os.makedirs(runlog_path, exist_ok=True)
    logging = init_logger(
        os.path.join(runlog_path, 'runlog.log')
    )
    tb_path = os.path.join(checkpoint_path, 'tensorboard')
    tb_logger = SummaryWriter(tb_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("***** Running evaluation *****")
    # We use recall@20 as the metric for saving the best model
    best_dev_recall = 0.0
    # early stop
    stop_step = 0
    stop_flag, update_flag = False, False

    all_checkpoint_path = opt.all_models_path
    best_checkpoint = os.path.join(all_checkpoint_path, 'best_dev')
    # add '/' character
    len_path = len(all_checkpoint_path) + 1
    id_file = os.path.join(all_checkpoint_path, 'checkpoint-*')
    len_file = len(glob.glob(id_file))
    logging.info(f"all checkpoint amounts : {len_file}")
    all_checkpoint = []
    for file in sorted(glob.glob(id_file), key=lambda name: int(name[len_path + 11:])):
        all_checkpoint.append(file)

    tokenizer = T5Tokenizer.from_pretrained(all_checkpoint[0])

    data_dir = os.path.join(opt.data_dir, opt.data_name)
    eval_file = os.path.join(data_dir, 'valid.txt')
    item_file = os.path.join(data_dir, 'item.txt')

    # We only use the title attribute in the Amazon dataset, and the title and address attributes in the Yelp dataset.
    if opt.data_name == 'beauty' or opt.data_name == 'sports' or opt.data_name == 'toys':
        item_desc = load_item_name(item_file)
    elif opt.data_name == 'yelp':
        item_desc = load_item_address(item_file)
    item_len = len(item_desc)
    logging.info(f"item len: {item_len}")
    eval_data = load_data(eval_file,item_desc)
    logging.info(f"dev len: {len(eval_data)}")
    item_data = load_item_data(item_desc)

    eval_seq_dataset = SequenceDataset(eval_data, tokenizer, opt)
    eval_seq_sampler = SequentialSampler(eval_seq_dataset)
    eval_seq_dataloader = DataLoader(
        eval_seq_dataset,
        sampler=eval_seq_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_seq_dataset.collect_fn
    )
    eval_item_dataset = ItemDataset(item_data, tokenizer, opt)
    eval_item_sampler = SequentialSampler(eval_item_dataset)
    eval_item_dataloader = DataLoader(
        eval_item_dataset,
        sampler=eval_item_sampler,
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_item_dataset.collect_fn
    )
    Ks = eval(opt.Ks)

    for i in range(len(all_checkpoint)):
        check_path = all_checkpoint[i]
        step = int(check_path[len_path + 11:])
        model_class = TASTEModel
        model = model_class.from_pretrained(check_path)
        model.to(device)
        metrics_dict = evaluate(model, eval_seq_dataloader, eval_item_dataloader, device, Ks, logging,
                                   tb_logger, step)
        cur_recall = metrics_dict[20]['recall']
        best_dev_recall, stop_step, stop_flag, update_flag = early_stopping(cur_recall, best_dev_recall, stop_step,
                                                                            opt.stopping_step)
        if update_flag:
            # Python version below 3.8 will report an error
            shutil.copytree(check_path, best_checkpoint, dirs_exist_ok=True)
            logging.info(
                'Saved Best:step:{},Recall@10：{:.4f}, Recall@20:{:.4f}, NDCG@10: {:.4f}, NDCG@20:{:.4f}'.format(
                    step,metrics_dict[10]['recall'], metrics_dict[20]['recall'],
                    metrics_dict[10]['ndcg'],
                    metrics_dict[20]['ndcg']))
        if stop_flag:
            logging.info(" Early stop!Finished!")
            break


    logging.info('***** Finish evaluation *****')


if __name__ == '__main__':
    main()