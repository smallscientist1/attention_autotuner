import torch
import time
import torch.nn.functional as F
from torch import nn, einsum
import os
import argparse
import torch.nn as nn
import ctypes

from autotuner.tunner import RetBwdTunner
import arch

import debugpy
# debugpy.connect(('localhost', 5678))
# torch.manual_seed(54)

dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='', help='the path to input data')
parser.add_argument('--bench_pytorch', action="store_true", default=False, help='')
parser.add_argument('--batch', type=int, default=4, help='')
parser.add_argument('--seqlen_q', type=int, default=2048, help='')
parser.add_argument('--seqlen_kv', type=int, default=2048, help='')
parser.add_argument('--nheads', type=int, default=8, help='')
parser.add_argument('--dim_qk', type=int, default=256, help='')
parser.add_argument('--dim_v', type=int, default=256, help='')
parser.add_argument('--operation', type=str, default='retnet_bwd', help='[retnet_bwd, attn_bwd]')
parser.add_argument('--arch', type=str, default="A100")
args = parser.parse_args()

batch = args.batch
seqlen_q = args.seqlen_q
seqlen_kv = args.seqlen_kv
nheads = args.nheads
dim_qk = args.dim_qk
dim_v = args.dim_v

if args.input_path != "":
    query = torch.load(os.path.join(args.input_path, "q.pt")).squeeze(1).cuda()[:batch, :nheads, :seqlen_q, :dim_qk]
    key = torch.load(os.path.join(args.input_path, "k.pt")).squeeze(1).cuda()[:batch, : nheads, :seqlen_kv, : dim_qk]
    value = torch.load(os.path.join(args.input_path, "v.pt")).squeeze(1).cuda()[:batch, :nheads, :seqlen_kv, :dim_v]
    mask = torch.load(os.path.join(args.input_path, "mask.pt")).cuda()[:nheads, :seqlen_q, :seqlen_kv]
    do = torch.load(os.path.join(args.input_path, "do.pt")).cuda()[:batch, :nheads, :seqlen_q, :dim_v]
    print(query.shape, key.shape, value.shape, mask.shape)
else:
    query = torch.randn([batch, nheads, seqlen_q, dim_qk], dtype=dtype, device='cuda:0')
    key = 3 * torch.randn([batch, nheads, seqlen_kv, dim_qk], dtype=dtype, device='cuda:0')
    value = 3 * torch.randn([batch, nheads, seqlen_kv, dim_v], dtype=dtype, device='cuda:0')
    mask = 0.5 * torch.randn([nheads, seqlen_q, seqlen_kv],dtype=dtype, device='cuda:0')
    # mask = torch.load(os.path.join(args.input_path, "mask.pt")).cuda()[:nheads, :seqlen_q, :seqlen_kv]
    do = torch.randn([batch, nheads, seqlen_q, dim_v], dtype=dtype, device='cuda:0')
    print(query.shape, key.shape, value.shape, mask.shape)


query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)

class RetAttention(nn.Module):
    def __init__(self):
        super(RetAttention, self).__init__()

    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-1, -2)
        qkm = qk * mask
        r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o = torch.matmul(qkm/r, v)  
        return o
    
softmax_scale = 0.125
class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-1, -2)
        attn = F.softmax(qk * softmax_scale, dim=-1)
        o = attn @ v
        return o

r = torch.zeros([batch, nheads, seqlen_q], dtype=torch.float32, device='cuda:0')
dq = torch.zeros_like(query)
dk = torch.zeros_like(key)
dv = torch.zeros_like(value)
dqaccum = torch.zeros_like(query, dtype=torch.float32)
if args.operation == "retnet_bwd":
    tunner = RetBwdTunner(arch=arch.__getattribute__(args.arch)(), torch_array=[query, key, value, mask, do,r,dq,dk,dv,dqaccum])
elif args.operation == "attn_bwd":
    # TODO
    tunner = None
tunner.tune(log_path="../../logs/")


torch_model = RetAttention() if args.operation == "retnet_bwd" else Attn()
torch_input = (query, key, value) if args.operation == "attn" else (query, key, value, mask)
if args.bench_pytorch:
    iters = 100
    output0 = torch_model(*torch_input)
    # warpup
    for i in range(iters):
        for x in torch_input:
            if isinstance(x, torch.Tensor):
                x.grad = None
        output0.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        for x in torch_input:
            if isinstance(x, torch.Tensor):
                x.grad = None
        output0.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    end = time.time()
    print("torch latency: ", (end - start) / iters * 1000, "ms")



