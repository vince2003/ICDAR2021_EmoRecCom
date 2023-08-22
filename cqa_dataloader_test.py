import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pdb
#from trial_mode import trial_arg
#------------config-------------------------
#import config as CONFIG
from config import get_config
args = get_config()
#----------------------addition------------------
#name_project, trial_ck, max_epochs, data_subset, shuffle, batch_size, resume=trial_arg()
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
        # if self.split == 'train':
        #     self.prep = config.train_transform
        # else:
        self.prep = config.test_transform

    def __len__(self):
        return len(self.cqa_data)

    def __getitem__(self, index):
        #pdb.set_trace()
        ques = torch.tensor(self.cqa_data[index]['embedded_transcript'])
        ans_bnr = torch.zeros((8,))
        ans_con = torch.zeros((8,))
        img_id = self.cqa_data[index]['img_id']
        img_path = os.path.join(self.config.root, self.config.dataset, 'images', self.split,
                                self.cqa_data[index]['img_id']+'.jpg')
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.prep(img)
        
        #pdb.set_trace()
        
        return ques, ans_bnr, ans_con, img_tensor, img_id





def collate_batch(data_batch):
    #pdb.set_trace()
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)




# %%
def build_dataloaders_test(config):
    #pdb.set_trace()

    test_datasets = []
    for split in config.test_filenames:
        cqa_test_data = json.load(open(os.path.join(config.root, config.dataset, 'qa', config.test_filenames[split])))
        # n = int(config.data_subset * len(cqa_test_data))
        # cqa_test_data = cqa_test_data[:n]
        test_datasets.append(CQADataset(cqa_test_data, split, config))

    test_dataloaders = []
    for tds in test_datasets:
        test_dataloaders.append(DataLoader(tds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch,
                                           num_workers=0, worker_init_fn=_init_fn))

    return test_dataloaders


def main():
    pass


if __name__ == '__main___':
    main()
