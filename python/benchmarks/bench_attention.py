import sys
# sys.path.append("../")
from do_bench import do_bench
from ops.attention_interface import flash_attn_func
import torch
import torch.nn.functional as F
import traceback

from arch import A100

dtype = torch.float16
device = torch.device("cuda")
device_type = A100()
softmax_scale = 0.125

CHECK_PYTORCH = True
BENCH_BWD = False
torch.manual_seed(0)

def is_close_my(a, a_ref, rtol=1e-3, atol=1e-3):
    try:
        torch.testing.assert_close(a, a_ref, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(e)
        # traceback.print_exc()
        traceback.print_stack()
        print("\n")
        return False

def benchmark_attention(batch, heads, seqlen_q, seqlen_kv, dim_qk, dim_v):
    q = torch.randn(batch, heads, seqlen_q, dim_qk, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_kv, dim_qk, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seqlen_kv, dim_v, device=device, dtype=dtype)
    if BENCH_BWD:
        q = q.requires_grad_()
        k = k.requires_grad_()
        v = v.requires_grad_()
        do = torch.randn(batch, heads, seqlen_q, dim_v, device=device, dtype=dtype)
    if CHECK_PYTORCH:
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn * softmax_scale, dim=-1)
        o_ref = attn @ v    

        o = flash_attn_func(q, k, v, device_type)

        is_close_my(o, o_ref, rtol=1e-3, atol=1e-3)

        if BENCH_BWD:
            o.backward(do)
            q_grad = torch.clone(q.grad)
            k_grad = torch.clone(k.grad)
            v_grad = torch.clone(v.grad)
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()

            o_ref.backward(do)
            q_grad_ref = q.grad
            k_grad_ref = k.grad
            v_grad_ref = v.grad
            is_close_my(q_grad, q_grad_ref, rtol=1e-3, atol=1e-3)
            is_close_my(k_grad, k_grad_ref, rtol=1e-3, atol=1e-3)
            is_close_my(v_grad, v_grad_ref, rtol=1e-3, atol=1e-3)
    results = do_bench(lambda: flash_attn_func(q,k,v,device_type), quantiles=[0.5, 0.2, 0.8])
    results_bwd = None
    if BENCH_BWD:
        o1 = flash_attn_func(q, k, v, device_type)
        results_bwd = do_bench(lambda: o1.backward(do, retain_graph=True), quantiles=[0.5, 0.2, 0.8])
    return results, results_bwd


batch = 4
heads = 8
seqlen_q = 2048
seqlen_kv = 2048
dim_qk = 256
dim_v = 256
res,res_bwd = benchmark_attention(batch, heads, seqlen_q, seqlen_kv, dim_qk, dim_v)
print("fwd: ",res)
if BENCH_BWD:
    print("bwd: ",res_bwd)
