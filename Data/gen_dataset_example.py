"""
We directly obtain the data set processed by Recbole from the DIF-SR project:https://github.com/AIM-SE/DIF-SR.
Here is a code example for obtaining the data. If necessary, please run it in the DIF-SR project.
We have provided the processed dataset.
########################
"""



# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""
import logging
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)

    if config['save_dataset']:
        dataset.save()

    logger.info(dataset)
    # save pickle
    # path_item = '{}_item.pickle'.format(dataset.dataset_name)
    # path_user = '{}_user.pickle'.format(dataset.dataset_name)
    # with open(path_item,'wb') as f:
    #     pickle.dump(dataset.field2token_id['item_id'],f)
    # with open(path_user, 'wb') as f:
    #     pickle.dump(dataset.field2token_id['session_id'],f)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    train_dict = train_data.dataset.inter_feat.numpy()
    valid_dict = valid_data.dataset.inter_feat.numpy()
    test_dict = test_data.dataset.inter_feat.numpy()

    """
    The following is the code for data processing.
    ########################
    """


    # Amazon

    # item.txt
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat['item_id']
    title_id = item_feat['title']
    catergory_id = item_feat['categories']
    brand_id = item_feat['brand']
    price = item_feat['price']
    rank = item_feat['sales_rank']

    title = dataset.field2id_token['title']
    category = dataset.field2id_token['categories']
    brand = dataset.field2id_token['brand']
    writer = open('item.txt', 'w', encoding='utf-8')
    writer.write('%s\t%s\t%s\t%s\t%s\t%s\n' % ('item_id', 'item_name','categories','brand','price','sales_rank'))
    i = 0
    for id, tid,cid,bid,p,r in zip(item_id, title_id,catergory_id,brand_id,price,rank):
        id = int(id)
        tid = int(tid)
        cid = int(cid)
        bid = int(bid)
        p = float(p)
        p = round(p,2)
        r = int(r)
        name = str(title[tid])
        cate = str(category[cid])
        bra = str(brand[bid])
        writer.write('%d\t%s\t%s\t%s\t%.2f\t%d\n' % (id, name,cate,bra,p,r))
        i += 1
    print('------------------finish---------------')
    print(i)


    # train.txt
    writer = open('train.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id','seq','target'))
    for user_id,seq_list,target in zip(train_dict['user_id'],train_dict['item_id_list'],train_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)

    # valid.txt
    writer = open('valid.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(valid_dict['user_id'], valid_dict['item_id_list'], valid_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)

    # test.txt
    writer = open('test.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(test_dict['user_id'], test_dict['item_id_list'], test_dict['item_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    print('------------------finish---------------')
    print(i)


    # yelp

    # item.txt
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat['business_id']
    title_id = item_feat['item_name']
    catergory_id = item_feat['categories']
    address_id = item_feat['address']
    city_id = item_feat['city']
    state_id = item_feat['state']
    title = dataset.field2id_token['item_name']
    category = dataset.field2id_token['categories']
    address = dataset.field2id_token['address']
    city = dataset.field2id_token['city']
    state = dataset.field2id_token['state']
    writer = open('item.txt', 'w', encoding='utf-8')
    writer.write('%s\t%s\t%s\t%s\t%s\t%s\n' % ('item_id', 'item_name','categories','address','city','state'))
    i = 0
    for id, tid,cid,cit,sid,aid in zip(item_id, title_id,catergory_id,city_id,state_id,address_id):
        id = int(id)
        tid = int(tid)
        cid = int(cid)
        cit = int(cit)
        sid = int(sid)
        aid = int(aid)
        name = str(title[tid])
        cate = str(category[cid])
        ccity = str(city[cit])
        sta = str(state[sid])
        add = str(address[aid])
        writer.write('%d\t%s\t%s\t%s\t%s\t%s\n' % (id, name,cate,add,ccity,sta))
        i += 1
    print('------------------finish---------------')
    print(i)



    # train.txt
    writer = open('train.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(train_dict['user_id'], train_dict['business_id_list'], train_dict['business_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    writer.close()
    print('------------------finish1---------------')
    print(i)

    # valid.txt
    writer = open('valid.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(valid_dict['user_id'], valid_dict['business_id_list'],
                                         valid_dict['business_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    writer.close()
    print('------------------finish2---------------')
    print(i)

    # test.txt
    writer = open('test.txt', 'w', encoding='utf-8')
    i = 0
    writer.write('%s\t%s\t%s\n' % ('user_id', 'seq', 'target'))
    for user_id, seq_list, target in zip(test_dict['user_id'], test_dict['business_id_list'],
                                         test_dict['business_id']):
        uid = int(user_id)
        writer.write('%d\t' % uid)
        for id in seq_list:
            writer.write('%d\t' % int(id))
        tid = int(target)
        writer.write('%d\n' % tid)
        i += 1
    writer.close()
    print('------------------finish3---------------')
    print(i)

    """
    Finish data processing.
    ########################
    """









    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)



    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }