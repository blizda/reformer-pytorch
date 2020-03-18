import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reformer_pytorch import Reformer, ReformerLM
from transformers import BertTokenizer, AdamW
import re
import os
from glob import glob
import json
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils

devices = xm.xla_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.max_len = 40960
model = ReformerLM(
    num_tokens= tokenizer.vocab_size,
    dim = 768,
    depth = 12,
    max_seq_len = tokenizer.max_len,
    heads = 48,
    lsh_dropout = 0.1,
    causal = False,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    weight_tie = False,   # tie parameters of each layer for no memory per additional depth
    attn_chunks = 8,        # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    twin_attention = False, # both branches of the reversible network will be attention
    use_full_attn = False,  # use full self attention, for comparison
    full_attn_thres = 128, # use full attention if context length is less than set value
    use_scale_norm = True,  # use scale norm from 'Transformers without tears' paper
    axial_position_emb = True,
    axial_position_shape = (640, 64),
    axial_position_dims = (384, 384)
)
model.train()
model.to(devices)
print("starting test")
inputs = torch.randint(low=0, high=tokenizer.vocab_size - 1, size=(10, tokenizer.max_len)).to(devices)
output = model(inputs)
print(output)
print("test pass")
