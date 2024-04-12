from autotuner.runtime import Runtime
from arch import Arch
from autotuner.configs import RetConfig, RetBwdConfig
import torch

from autotuner.tunner import RetTunner

class RetNetAttn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask, device_type: Arch, causal = False):

        assert q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16 and mask.dtype == torch.float16
        q, k, v = [x.contiguous() for x in [q, k, v]]
        maskr = mask.detach().contiguous()
        batch_size, nheads, seqlen_q, key_dim = q.shape
        seqlen_k = k.shape[2]
        head_dim = v.shape[-1]

        assert head_dim % 64 == 0 and head_dim <=512
        assert key_dim % 64 == 0 and key_dim <= 256
        
        o = torch.zeros(batch_size, nheads, seqlen_q, head_dim, device=q.device, dtype=torch.float16)
        r = torch.zeros(batch_size, nheads, seqlen_q, device=q.device, dtype=torch.float32)
        
        cc = RetTunner(arch=device_type, torch_array=[q, k, v, maskr, o, r]).tune(log_path="../../logs/")
        # cc = RetConfig(Br=128, Bc=128, Kd=key_dim, D = head_dim, unrollLastIter=0, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
        # cc.set_fuse_type("register")
        # TODO: torch.cuda.get_device_properties(0)
        Runtime(device_type, cc, tmp_dir="../../tmp/ret").apply([q, k, v, maskr, o, r])
        ctx.save_for_backward(q, k, v, mask, r)
        ctx.device_type = device_type
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, mask, r = ctx.saved_tensors
        device_type = ctx.device_type
        batch_size, nheads, seqlen_q, key_dim = q.shape
        seqlen_k = k.shape[2]
        head_dim = v.shape[-1]

        do = do.contiguous()
        mask = mask.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        dq_accum = torch.zeros([batch_size, nheads, seqlen_q, key_dim], device=q.device, dtype=torch.float32)

        cc = RetBwdConfig(Br=64, Bc=64, Kd=key_dim, D=head_dim, mmawarpsN=2, mmawarpsN_dk=4, mmawarpsN_dv=4, mmawarpsN_dq=4)
        Runtime(device_type,cc , tmp_dir="../../tmp/ret_bwd").apply([q, k, v, mask, do, r, dq, dk, dv, dq_accum])
        return dq, dk, dv, None, None

RetNetAttnFunc = RetNetAttn.apply