from autotuner.runtime import Runtime
from arch import Arch
from autotuner.configs import RetConfig, RetBwdConfig
import torch

from autotuner.tunner import AttnTunner

class MultiHeadAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, device_type: Arch, causal = False):
        """
            q: (batch_size, nheads, seqlen_q, key_dim)
            k: (batch_size, nheads, seqlen_k, key_dim)
            v: (batch_size, nheads, seqlen_v, head_dim)

        """
        q, k, v = [x.contiguous() for x in [q, k, v]]
        batch_size, nheads, seqlen_q, key_dim = q.shape
        seqlen_k = k.shape[2]
        head_dim = v.shape[-1]

        assert head_dim % 64 == 0 and head_dim <=512
        assert key_dim % 64 == 0 and key_dim <= 256

        o = torch.zeros(batch_size, nheads, seqlen_q, head_dim, device=q.device, dtype=q.dtype)
        r = torch.zeros(batch_size, nheads, seqlen_q, device=q.device, dtype=torch.float32)

        cc = AttnTunner(arch=device_type, torch_array=[q, k, v, o]).tune(log_path="../../logs/")
        Runtime(device_type, cc, tmp_dir="../../tmp/attn").apply([q, k, v, o])
        ctx.save_for_backward(q, k, v) # , lse)
        ctx.device_type = device_type
        return o

    @staticmethod
    def backward(ctx, do):
        pass
        # q, k, v, mask, d = ctx.saved_tensors
        # # q, k, v, mask= ctx.saved_tensors
        # Br = 64
        # Bc = 64
        # batch_size, nheads, seqlen_q, key_dim = q.shape
        # seqlen_k = k.shape[2]
        # Tr = seqlen_q // Br
        # Tc = seqlen_k // Bc
        # maskr = mask.detach().view(nheads, Tr, Br, Tc, Bc).permute(0, 3, 1, 2, 4).contiguous()
        # dq = torch.zeros_like(q)
        # dk = torch.zeros_like(k)
        # dv = torch.zeros_like(v)
        # lib = ctypes.cdll.LoadLibrary(BWD_LIB_PATH)
        # lib.kernel_entry.argstypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        # ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]   
        # lib.kernel_entry.restype = ctypes.c_int32
        # torch_arrs = [q, k, v, maskr, do, d, dq, dk, dv]
        # stats = lib.kernel_entry(*[ctypes.cast(arr.data_ptr(), ctypes.c_void_p) for arr in torch_arrs])
        # return dq, dk, dv, None
    
flash_attn_func = MultiHeadAttnFunc.apply
