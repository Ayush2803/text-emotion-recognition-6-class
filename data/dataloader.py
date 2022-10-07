import os
import logging
import pickle
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer
from transformers import RobertaTokenizer


__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

def get_emotion2id(DATASET="IEMOCAP"):
    """Get a dict that converts string class to numbers."""
    if DATASET == "IEMOCAP":
        # IEMOCAP originally has 11 classes but we'll only use 6 of them.
        emotions = [
            "neutral",
            "frustration",
            "sadness",
            "anger",
            "excited",
            "happiness",
        ]
        emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
        id2emotion = {val: key for key, val in emotion2id.items()}

    return emotion2id, id2emotion

class MMDataset(Dataset):
  def __init__(self, data_dir=os.path.join(os.getcwd(),'data/'), num_classes=6,past=2, split='train'):
    self.root_dir=data_dir
    self.past=past
    self.split=split
    self.raw_text_dir = os.path.join(self.root_dir, 'raw-texts', self.split)
    self.emotion2id, self.id2emotion = get_emotion2id()
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    self.inputs=[]
    fname = self.root_dir+"/data-"+ str(num_classes)+ "-" + str(past) + "/" + self.split+'.pkl'
    with open(fname, 'rb') as readfile:
      self.inputs=pickle.load(readfile)
    
  
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self,index):
    text=self.inputs[index]['text']
    label=self.inputs[index]['label']
    text_tokens = self.tokenizer(text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length', return_token_type_ids=True,
                                     return_tensors="pt")
    return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label

def MMDataLoader(args):

    train_set = MMDataset(num_classes= args.num_classes, past= args.past, split='train')
    valid_set = MMDataset(num_classes= args.num_classes, past= args.past, split='val')
    test_set = MMDataset(num_classes= args.num_classes, past= args.past, split='test')
    print("Train Dataset: ", len(train_set))
    print("Valid Dataset: ", len(valid_set))
    print("Test Dataset: ", len(test_set))

    # print(args.num_workers, args.batch_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, pin_memory=False, drop_last=True)
    valid_loader = DataLoader(valid_set,  batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=False, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=False, drop_last=True)

    return train_loader, valid_loader, test_loader
