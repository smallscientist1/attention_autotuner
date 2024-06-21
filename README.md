# attention_autotuner
An autotuner for the flash version of **attention** and **retnet**

## Introduction

This project provide an autotuner for the flash version of **attention** and **retnet**. Users can use it as an pytorch func.

```python
# Attention on A100
from ops.attention_interface import flash_attn_func
import torch
from arch import A100

device_type = A100()
dtype = torch.float16
device = torch.device("cuda")

q = torch.randn(batch, heads, seqlen_q, dim_qk, device=device, dtype=dtype)
k = torch.randn(batch, heads, seqlen_kv, dim_qk, device=device, dtype=dtype)
v = torch.randn(batch, heads, seqlen_kv, dim_v, device=device, dtype=dtype)

o = flash_attn_func(q,k,v,device_type)
    
```

```python
# retnet on RTX4090
from ops.retnet_interface import RetNetAttnFunc
import torch

device_type = RTX4090()
dtype = torch.float16
device = torch.device("cuda")

q = torch.randn(batch, heads, seqlen_q, dim_qk, device=device, dtype=dtype)
k = torch.randn(batch, heads, seqlen_kv, dim_qk, device=device, dtype=dtype)
v = torch.randn(batch, heads, seqlen_kv, dim_v, device=device, dtype=dtype)
mask = torch.randn(heads, seqlen_q, seqlen_kv, device=device, dtype=dtype)

o = RetNetAttnFunc(q, k, v, mask, device_type)

do = torch.randn(batch, heads, seqlen_q, dim_v, device=device, dtype=dtype)
o.backward(do)

```


## Installation

### Requirements
- cuda 12.3
- cmake 3.24

### To install

- clone this repo and its submodule [cutlass](https://github.com/NVIDIA/cutlass.git)
```
git clone --recursive https://github.com/smallscientist1/attention_autotuner.git
```
- add to PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:/path/to/attention_autotuner/python
```

## Benchmark

- build the C++ benchmark on nvidia Ampere GPU(e.g. A100)
```
cd benchmarks
mkdir build
cd build
cmake -DPROJECT_CUDA_ARCH="80" ..
```

## Performance



## Appendix

### attention algo

#### flash attention
- q @ k
- reduce_max(qk)
- scale = exp(m_old-m_new)
- lse* scale
- acco * scale
- accs * exp(accs-m_new)
- lse = reduce_sum(accs)

#### retnet parallel
- q @ k
- qk * mask
- reduce_abs(qk)
- clamp(r)
- scale = r_old/r_new
- acco * scale
- accs / r_new


## TODO
- chunkwise retnet
- cost model
- autotuner(more general policy for retnet)
- elementwise op
- attention backward
- retnet performance issue(added load q once, mask stage 2?)
- causal config
- retnet parallel scan version seqlen_q != seqlen_kv
- retnet parallel scan template
- retnet bwd load_q_once,causal
- the performance of python interface?
- retnet backward num_stage_qk=2 bug
- retnet fwd regfuse d=192 bug
