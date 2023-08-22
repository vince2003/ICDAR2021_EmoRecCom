import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pdb
from trial_mode import trial_arg

#----------------------addition------------------
name_project, trial_ck, max_epochs, data_subset, shuffle, batch_size, resume=trial_arg()
def _init_fn(worker_id):
    np.random.seed(0)
    
    
class CQADataset(Dataset):
    def __init__(self, cqa_data, split, config):
        self.cqa_data = cqa_data
        #self.ques2idx = ques2idx
        #self.ans2idx = ans2idx
        #self.maxlen = maxlen
        self.split = split
        self.config = config
        if self.split == 'train':
            self.prep = config.train_transform
        else:
            self.prep = config.test_transform

    def __len__(self):
        return len(self.cqa_data)

    def __getitem__(self, index):
        #pdb.set_trace()
        ques = torch.tensor(self.cqa_data[index]['embedded_transcript'])
        #pdb.set_trace()
        if 'test' not in self.split:
            #ans = encode_answers(self.cqa_data[index]['answer'], self.config)
            if 'train' in self.split:
                #if self.config.loss=="BCE":
                ans_bnr = torch.tensor(list(map(float, self.cqa_data[index]['answer_binary'].split(','))))
                #else:
                ans_con = torch.tensor(list(map(float, self.cqa_data[index]['answer_continuous'].split(','))))
            else:
                ans_bnr = torch.tensor(list(map(float, self.cqa_data[index]['answer_binary'].split(','))))
            #pdb.set_trace()
                ans_con = torch.zeros((1,))
            #ans_metric=torch.tensor(list(map(float, self.cqa_data[index]['answer_binary'].split(','))))
        else:
            ans_bnr = torch.zeros((8,))
            ans_con = torch.zeros((8,))
        img_id = self.cqa_data[index]['img_id']
        img_path = os.path.join(self.config.root, self.config.dataset, 'images', self.split,
                                self.cqa_data[index]['img_id']+'.jpg')
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.prep(img)
        
        #pdb.set_trace()
        
        return ques, ans_bnr, ans_con, img_tensor, img_id


def encode_transcript(transcript, ques2idx, maxlen):
    
    ques_vec = torch.zeros(maxlen).long()
    ques_words = word_tokenize(transcript.lower())
    ques_len = len(ques_words)
    for i, word in enumerate(ques_words):
        ques_vec[i] = ques2idx.get(word, len(ques2idx))  # last idx is reserved for <UNK>, needed for real OCRs
    return ques_vec, ques_len


# def encode_answers(answer, config):
#     if config.dataset == 'FigureQA':

        
#         label=list(map(int, answer.split(',')))

#         return label
    # else:
    #     return ans2idx.get(answer, len(ans2idx))


def collate_batch(data_batch):
    #pdb.set_trace()
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)


def tokenize(q):
    return word_tokenize(q['transcript'].lower())


def build_lut(cqa_train_data):
    print("Building lookup table for transcript and answer tokens")
    #pdb.set_trace()

    pool = ProcessPoolExecutor(max_workers=8)
    transcript = list(pool.map(tokenize, cqa_train_data, chunksize=1000))
    pool.shutdown()
    print("Finished")

    maxlen = max([len(q) for q in transcript])
    unique_tokens = set([t for q in transcript for t in q])
    ques2idx = {word: idx + 1 for idx, word in enumerate(unique_tokens)}  # save 0 for padding
    
    # answers = set([q['answer'] for q in cqa_train_data])
    #ans2idx = {ans: idx for idx, ans in enumerate(answers)}
    #ans2idx={"1": [0,1], "0": 1}
    #pdb.set_trace()
    return ques2idx, maxlen


# %%
def build_dataloaders(config):
    #pdb.set_trace()
    
    cqa_train_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.train_filename)))
    # if config.lut_location == '':
    #     ques2idx, maxlen = build_lut(cqa_train_data)
    # else:
    #     lut = json.load(open(config.lut_location, 'r'))
    #     #ans2idx = lut['ans2idx']
    #     ques2idx = lut['ques2idx']
    #     maxlen = lut['maxlen']

    m = int(data_subset * len(cqa_train_data))
    #np.random.seed(666)
    #np.random.shuffle(cqa_train_data)
    cqa_train_data = cqa_train_data[:m]
    train_dataset = CQADataset(cqa_train_data, 'train', config)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                                  num_workers=0, worker_init_fn=_init_fn)
    
    

    val_datasets = []
    for split in config.val_filenames:        
        cqa_val_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.val_filenames[split])))
        val_datasets.append(CQADataset(cqa_val_data, split, config))
        #pdb.set_trace()

    val_dataloaders = []
    for vds in val_datasets:
        val_dataloaders.append(DataLoader(vds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
                                          num_workers=0, worker_init_fn=_init_fn))

    test_datasets = []
    for split in config.test_filenames:
        cqa_test_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.test_filenames[split])))
        # n = int(config.data_subset * len(cqa_test_data))
        # cqa_test_data = cqa_test_data[:n]
        test_datasets.append(CQADataset(cqa_test_data, split, config))

    test_dataloaders = []
    for tds in test_datasets:
        test_dataloaders.append(DataLoader(tds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch,
                                           num_workers=0, worker_init_fn=_init_fn))

    return train_dataloader, val_dataloaders, test_dataloaders


def main():
    pass


if __name__ == '__main___':
    main()
