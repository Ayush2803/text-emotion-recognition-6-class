import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import RobertaModel

__all__ = ['WMSA']


class WMSA(nn.Module):
    def __init__(self, args):
        super(WMSA, self).__init__()
        # text subnets
        self.args = args
        self.text_model = RobertaModel.from_pretrained('roberta-large')
        self.attn_layer = nn.MultiheadAttention(embed_dim= self.args.text_appended, num_heads=self.args.attn_heads)
        self.post_text_dropout = nn.Dropout(p=0)
        self.post_text_layer_1 = nn.Linear(
            self.args.text_appended, self.args.post_layer_dim)
        self.post_text_layer_2 = nn.Linear(self.args.post_layer_dim, args.post_dim)
        self.post_text_layer_3 = nn.Linear(args.post_dim, args.output_dim)

    def forward(self, text=None, audio=None, video=None, label=None):

        input_ids = torch.squeeze(text[0], 1)
        input_mask = torch.squeeze(text[2], 1)
        segment_ids = torch.squeeze(text[1], 1)

        text = self.text_model(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0][:, 0, :]
        zero = torch.zeros(size=(self.args.batch_size,1)).to(self.args.device)
        text= torch.concat((zero,text,zero), axis=1)
        x, _ = self.attn_layer(text,text,text)
        # text
        text_d = self.post_text_dropout(x)

        # text_d = self.post_text_dropout(text)

        text_h = F.relu(self.post_text_layer_1(
            text_d), inplace=False)  # (32, 128)
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)
        return output_text
