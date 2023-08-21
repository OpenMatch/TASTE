import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self,data,tokenizer,args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.template = self.toke_template()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    def toke_template(self):
        t_list = []
        template1 = "Here is the visit history list of user: "
        template2 = " recommend next item "
        t1 = self.tokenizer.encode(template1, add_special_tokens=False,
                                     truncation=False)
        t_list.append(t1)
        t2 = self.tokenizer.encode(template2, add_special_tokens=False,
                                   truncation=False)
        t_list.append(t2)
        return t_list
    def collect_fn(self, data):
        sequence_ids = []
        sequence_masks = []
        batch_target = []
        for example in data:
            batch_target.append(example[1])
            seq_text = example[0]
            seq = self.tokenizer.encode(seq_text, add_special_tokens=False,
                                        truncation=False)
            # Segment the user sequence text first, then add prompt to the first subsequence
            seq = list_split(seq,self.args.split_num)
            seq[0] = self.template[0] + seq[0] + self.template[1]

            s_ids = []
            s_masks = []
            for s in seq:
                outputs = self.tokenizer.encode_plus(
                    s,
                    max_length=self.args.seq_size,
                    pad_to_max_length=True,
                    return_tensors='pt',
                    truncation=True,
                )
                input_ids = outputs["input_ids"]
                attention_mask = outputs["attention_mask"]
                s_ids.append(input_ids)
                s_masks.append(attention_mask)
            s_ids = torch.cat(s_ids, dim=0)
            s_masks = torch.cat(s_masks, dim=0)
            cur_item = s_ids.size(0)
            if cur_item < self.args.num_passage:
                b = self.args.num_passage - cur_item
                l = s_ids.size(1)
                pad = torch.zeros([b, l], dtype=s_ids.dtype)
                s_ids = torch.cat((s_ids, pad), dim=0)
                s_masks = torch.cat((s_masks, pad), dim=0)
            sequence_ids.append(s_ids[None])
            sequence_masks.append(s_masks[None])

        sequence_ids = torch.cat(sequence_ids, dim=0)
        sequence_masks = torch.cat(sequence_masks, dim=0)

        return {
            "seq_ids": sequence_ids,
            "seq_masks": sequence_masks,
            "target_list": batch_target,
        }



class ItemDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, batch):

        item_ids, item_masks = encode_batch(batch, self.tokenizer, self.args.item_size)

        return {
            "item_ids": item_ids,
            "item_masks": item_masks,
        }


def list_split(array,n):
    split_list = []
    s1 = array[:n]
    s2 = array[n:]
    split_list.append(s1)
    if len(s2) != 0:
        split_list.append(s2)
    return split_list




def encode_batch(batch_text, tokenizer, max_length):
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True,
    )
    input_ids = outputs["input_ids"]
    input_ids = torch.unsqueeze(input_ids, 1)
    attention_mask = outputs["attention_mask"]
    attention_mask = torch.unsqueeze(attention_mask, 1)

    return input_ids, attention_mask




def load_item_name(filename):
    # load name
    item_desc = dict()
    id_prefix = 'id:'
    title_prefix = 'title:'
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        item_id = int(line[0])
        name = line[1]
        name = name.replace('&amp;', '')
        item_text = id_prefix + " " + str(item_id) + " " + title_prefix + " " + name
        item_desc[item_id] = item_text
    return item_desc


def load_item_address(filename):
    # load name and address
    item_desc = dict()
    id_prefix = 'id:'
    title_prefix = 'title:'
    passage_prefix = 'address:'
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        item_id = int(line[0])
        name = line[1]
        address = line[3]
        city = line[4]
        state = line[5]
        item_text = id_prefix + " " + str(item_id) + " " + title_prefix + " " + name + " " + \
                    passage_prefix + " " + address + " " + city + " " + state
        item_desc[item_id] = item_text
    return item_desc






def load_data(filename,item_desc):
    data = []
    lines = open(filename, 'r').readlines()
    for line in lines[1:]:
        example = list()
        line = line.strip().split('\t')
        target = int(line[-1])
        seq_id = line[1:-1]
        text_list = []
        for id in seq_id:
            id = int(id)
            if id==0:
                break
            text_list.append(item_desc[id])
        text_list.reverse()
        seq_text = ', '.join(text_list)
        example.append(seq_text)
        example.append(target)
        data.append(example)

    return data



def load_item_data(item_desc):
    data = []
    keys = item_desc.keys()
    for i in keys:
        text = item_desc[i]
        data.append(text)

    return data