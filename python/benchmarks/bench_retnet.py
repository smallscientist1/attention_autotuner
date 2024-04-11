from do_bench import do_bench
import sys
# sys.path.append("/home/v-feiychen/attention_autotuner/python")
from ops.retnet_interface import RetNetAttnFunc
import torch

from arch import A100

dtype = torch.float16
device = torch.device("cuda")
device_type = A100()

CHECK_PYTORCH = True

def benchmark_retnet(batch, heads, seqlen_q, seqlen_kv, dim_qk, dim_v):
    q = torch.randn(batch, heads, seqlen_q, dim_qk, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_kv, dim_qk, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seqlen_kv, dim_v, device=device, dtype=dtype)
    mask = torch.randn(heads, seqlen_q, seqlen_kv, device=device, dtype=dtype)
    if CHECK_PYTORCH:
        attn = q @ k.transpose(-1, -2)
        qkm = attn * mask
        r_ref = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        o_ref = (qkm/r_ref) @ v

        o = RetNetAttnFunc(q, k, v, mask, device_type)
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    results = do_bench(lambda: RetNetAttnFunc(q,k,v,mask,device_type), quantiles=[0.5, 0.2, 0.8])
    return results


batch = 4
heads = 8
seqlen_q = 2048
seqlen_kv = 2048
dim_qk = 256
dim_v = 256
res = benchmark_retnet(batch, heads, seqlen_q, seqlen_kv, dim_qk, dim_v)
print(res)