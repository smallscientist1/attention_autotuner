import torch
import time
import torch.nn.functional as F
from torch import nn, einsum
import os
import argparse
import torch.nn as nn
import ctypes
from do_bench import do_bench

# torch.manual_seed(54)

dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument('--test_bwd', action="store_true", default=False, help='')
parser.add_argument('--check_pytorch', action="store_true", default=False, help='')
parser.add_argument('--batch', type=int, default=4, help='')
parser.add_argument('--seqlen_q', type=int, default=2048, help='')
parser.add_argument('--seqlen_kv', type=int, default=2048, help='')
parser.add_argument('--nheads', type=int, default=8, help='')
parser.add_argument('--dim_qk', type=int, default=256, help='')
parser.add_argument('--dim_v', type=int, default=256, help='')
parser.add_argument('--iters', type=int, default=1, help='')
parser.add_argument('--womask', action="store_true", default=False, help='')
parser.add_argument('--save_onnx', action="store_true", default=False, help='')
parser.add_argument('--flash_kernel', type=str, default='', help='[template, triton]')
parser.add_argument('--operation', type=str, default='retnet', help='[retnet, attn]') # retnet:causal; attn: non-causal!!!
parser.add_argument('--arch', type=str, default="A100")
args = parser.parse_args()

batch = args.batch
seqlen_q = args.seqlen_q
seqlen_kv = args.seqlen_kv
nheads = args.nheads
dim_qk = args.dim_qk
dim_v = args.dim_v
iters = args.iters


softmax_scale = 0.125
class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-1, -2)
        attn = F.softmax(qk * softmax_scale, dim=-1)
        o = attn @ v
        return o

query = torch.randn([batch, nheads, seqlen_q, dim_qk], dtype=dtype, device='cuda:0')
key = torch.randn([batch, nheads, seqlen_kv, dim_qk], dtype=dtype, device='cuda:0')
value = torch.randn([batch, nheads, seqlen_kv, dim_v], dtype=dtype, device='cuda:0')

tt = do_bench(lambda: Attn()(query, key, value), quantiles=[0.5, 0.2, 0.8])
print(tt)
