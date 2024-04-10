import torch
import time
import torch.nn.functional as F
from torch import nn, einsum
import os
import argparse
import torch.nn as nn
import ctypes

from tunner import RetTunner, AttnTunner
import arch

# torch.manual_seed(54)

dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='', help='the path to input data')
parser.add_argument('--test_bwd', action="store_true", default=False, help='')
parser.add_argument('--bench_pytorch', action="store_true", default=False, help='')
parser.add_argument('--batch', type=int, default=4, help='')
parser.add_argument('--seqlen_q', type=int, default=2048, help='')
parser.add_argument('--seqlen_kv', type=int, default=2048, help='')
parser.add_argument('--nheads', type=int, default=8, help='')
parser.add_argument('--dim_qk', type=int, default=256, help='')
parser.add_argument('--dim_v', type=int, default=256, help='')
parser.add_argument('--operation', type=str, default='attn', help='[retnet, attn]')
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
    print(query.shape, key.shape, value.shape, mask.shape)
else:
    query = torch.randn([batch, nheads, seqlen_q, dim_qk], dtype=dtype, device='cuda:0')
    key = 3 * torch.randn([batch, nheads, seqlen_kv, dim_qk], dtype=dtype, device='cuda:0')
    value = 3 * torch.randn([batch, nheads, seqlen_kv, dim_v], dtype=dtype, device='cuda:0')
    mask = 0.5 * torch.randn([nheads, seqlen_q, seqlen_kv],dtype=dtype, device='cuda:0')
    # mask = torch.load(os.path.join(args.input_path, "mask.pt")).cuda()[:nheads, :seqlen_q, :seqlen_kv]
    print(query.shape, key.shape, value.shape, mask.shape)

if args.test_bwd:
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    grad = torch.randn([batch, nheads, seqlen_q, dim_v], dtype=dtype, device='cuda:0')
    # grad = torch.load(os.path.join(args.input_path, "do.pt")).cuda()[:batch, :nheads, :seqlen_q, :dim_v]

'''
Original:
q[B, H, Q, Kd]
k[B, H, K, Kd]
v[B, H, K, D]
------------
qk  = q @ k.transpose(-1, -2)
m = qk.max(dim=-1)
p = exp(qk - m)
s = p / p.sum(dim=-1)
o = s@v
-------------

Flash:
q[B, H, Br, Kd]
k[B, H, Bc, Kd]
v[B, H, Bc, D]

m_new[B, H, Br]
lse_new[B, H, Br]
acco[B, H, Br, D]
----------------------
m = -inf
m_new = -inf
lse_new = -inf
acco = 0
for (int j = 0; j < K/Bc; j ++)
{
  qk = q @ k.transpose(-1, -2)
  m_new = max(qk.max(dim=-1),m_new)
  p = exp(qk - m_new)
  lse_new = m_new + log(exp(lse_new-m_new) + p.sum(dim=-1))
  acco = acco * exp(m - m_new)
  m=m_new
  acco += p@v
}
out = acco * exp(m_new - lse_new)
-------------------------
'''
softmax_scale = 0.125
class FlashAttn(nn.Module):
    def __init__(self):
        super(FlashAttn, self).__init__()

    def forward(self, q, k, v):
        # TODO: FlashAttn for onnx
        qk = q @ k.transpose(-1, -2)
        m,_ = qk.max(dim=-1, keepdim=True)
        p = torch.exp(qk - m)
        lse = m + torch.log(0 + p.sum(dim=-1, keepdim=True))
        o = torch.matmul(p, v)  
        return o

class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, q, k, v):
        qk = q @ k.transpose(-1, -2)
        attn = F.softmax(qk * softmax_scale, dim=-1)
        o = attn @ v
        return o

'''
Original:
q[B, H, Q, Kd]
k[B, H, K, Kd]
v[B, H, K, D]
mask[H, Q, K]
------------
qk  = q @ k.transpose(-1, -2)
qkm = qk * m
r = qkm.detach().abs().sum(dim=-1).clamp(min=1)
s = qkm/r
o = s@v
-------------

Flash:
q[B, H, Br, Kd]
k[B, H, Bc, Kd]
v[B, H, Bc, D]
mask[H, Br, Bc]

r[B, H, Br]
acco[B, H, Br, D]
----------------------
r_new = 0
r_wo_clamp = 0
acco = 0
for (int j = 0; j < K/Bc; j ++)
{
  qkm = (q@k.transpose(-1,-2)) * m
  r_wo_clamp += qkm.detach().abs().sum(dim=-1)
  r_new = max(r_wo_clamp, 1)
  if (j != 0)
  { 
      acco = acco * r / r_new
      r = r_new
  }
  acco += (qkm/r_new)@v
}
-------------------------
'''
class RetAttention(nn.Module):
    def __init__(self):
        super(RetAttention, self).__init__()

    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-1, -2)
        qkm = qk * mask
        r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o = torch.matmul(qkm/r, v)  
        return o
    
class FlashRetAttention(nn.Module):
    def __init__(self):
        super(FlashRetAttention, self).__init__()

    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-1, -2)
        qkm = qk * mask
        r_wo_clamp = qkm.detach().abs().sum(dim=-1, keepdim=True)
        r_new = r_wo_clamp.clamp(min=1.0)
        o = torch.matmul(qkm/r_new, v)  
        return o


output = torch.zeros([batch, nheads, seqlen_q, dim_v], dtype=dtype, device='cuda:0')
r = torch.zeros([batch, nheads, seqlen_q], dtype=torch.float32, device='cuda:0')
if args.operation == "retnet":
    tunner = RetTunner(arch=arch.__getattribute__(args.arch)(), torch_array=[query, key, value, mask, output,r])
elif args.operation == "attn":
    tunner = AttnTunner(arch=arch.__getattribute__(args.arch)(), torch_array=[query, key, value, output])
tunner.tune()


torch_model = RetAttention() if args.operation == "retnet" else Attn()
torch_input = (query, key, value) if args.operation == "attn" else (query, key, value, mask)
if args.bench_pytorch:
    iters = 100
    # warpup
    for i in range(iters):
        output0 = torch_model(*torch_input)
    torch_output = torch_model(*torch_input)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(iters):
        output0 = torch_model(*torch_input)
    torch.cuda.synchronize()
    end = time.time()
    print("torch latency: ", (end - start) / iters * 1000, "ms")



