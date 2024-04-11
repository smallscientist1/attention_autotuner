import torch
from autotuner.runtime import Runtime

if __name__ == "__main__":
    torch.manual_seed(0)
    b = 4
    h = 4
    seq_q = 2048
    seq_kv = 2048
    Kd = 256
    D = 256
    q = torch.randn([b, h, seq_q, Kd], device="cuda:0", dtype=torch.float16)
    k = torch.randn([b, h, seq_kv, Kd], device="cuda:0", dtype=torch.float16)
    v = torch.randn([b, h, seq_kv, D], device="cuda:0", dtype=torch.float16)
    mask = torch.randn([h, seq_q, seq_kv], device="cuda:0", dtype=torch.float16)
    r = torch.zeros([b, h, seq_q], device="cuda:0", dtype=torch.float32)
    o = torch.zeros([b, h, seq_q, D], device="cuda:0", dtype=torch.float16)
    
    from arch import A100, RTX4090
    '''
    from config import AttnConfig

    cc = AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
    cc.set_fuse_type("register")
    # cc = AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
    # result error!
    cc = AttnConfig(Br=64, Bc=64, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=64, num_stages_v=1, Nthreads=256)
    # cc = AttnConfig(Br=128, Bc=32, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=1, Nthreads=256)
    # cc = AttnConfig(Br=64, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
    cc.set_fuse_type("shared")

    Runtime(A100(), cc,tmp_dir="../tmp/attn").apply([q, k, v, o])

    import torch.nn.functional as F
    softmax_scale = 0.125
    attn = q @ k.transpose(-1, -2)
    o_ref = F.softmax(attn * softmax_scale, dim=-1) @ v

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    '''

    # '''
    from autotuner.configs import RetConfig
    # cc = RetConfig(Br=128, Bc = 128, Kd = 256, D = 256, BlockKSmem=256, BlockKSmem2=32, num_stages_qk=1, num_stages_mask=1, num_stages_v=2, Nthreads=256, unrollLastIter=1)
    # result error!
    cc = RetConfig(Br=128, Bc = 128, Kd = 256, D = 256, BlockKSmem=64, BlockKSmem2=128, num_stages_qk=2, num_stages_mask=1, num_stages_v=1, Nthreads=256, unrollLastIter=0)
    cc.set_fuse_type("register")

    # cc = RetConfig(Br=128, Bc = 128, Kd = 256, D = 256, BlockKSmem=64, BlockKSmem2=128, num_stages_qk=2, num_stages_mask=1, num_stages_v=1, Nthreads=256, unrollLastIter=0)
    # result error!
    cc = RetConfig(Br=64, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, num_stages_mask=1, Nthreads=256)
    cc.set_fuse_type("shared")

    Runtime(A100(), cc,tmp_dir="../../tmp/ret").apply([q, k, v, mask, o, r])

    import torch.nn.functional as F
    attn = q @ k.transpose(-1, -2)
    qkm = attn * mask
    r_ref = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o_ref = (qkm/r_ref) @ v

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(r, r_ref.squeeze(-1).float(), rtol=1e-3, atol=1e-3)
    # '''

    '''
    from config import AttnConfig

    cc = AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=0, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
    cc.set_fuse_type("register")

    Runtime(RTX4090(), cc,tmp_dir="../tmp/attn").apply([q, k, v, o])

    import torch.nn.functional as F
    softmax_scale = 0.125
    attn = q @ k.transpose(-1, -2)
    o_ref = F.softmax(attn * softmax_scale, dim=-1) @ v

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    '''
