from autotuner.tunner import AttnTunner
from ncu_profile import ncu_cycles
import pprint

import arch
import torch
arch_type = "RTX4090" if torch.cuda.get_device_capability(0) == (8,9) else "A100" if torch.cuda.get_device_capability(0) == (8,0) else None

batch = 4
seqlen_q = 2048
seqlen_kv = 2048
nheads = 8
dim_qk = 128
dim_v = 128

if __name__ == "__main__":
    q = torch.randn([batch, nheads, seqlen_q, dim_qk], dtype=torch.float16, device='cuda:0')
    k = torch.randn([batch, nheads, seqlen_kv, dim_qk], dtype=torch.float16, device='cuda:0')
    v = torch.randn([batch, nheads, seqlen_kv, dim_v], dtype=torch.float16, device='cuda:0')
    o = torch.zeros([batch, nheads, seqlen_q, dim_v], dtype=torch.float16, device='cuda:0')

    
    tunner = AttnTunner(arch=arch.__getattribute__(arch_type)(), torch_array=[q,k,v,o])
    configs = tunner.get_tuned_configs()
    cost_model_configs = []
    for config in configs:
        Nwarps = config.Nthreads // 32

        cost_model_config = (
            (seqlen_q, seqlen_kv, dim_qk, dim_v),
            (config.Br, config.Bc, config.BlockKSmem, config.D),
            (config.Br//(Nwarps//config.warps_mma1_n), config.Bc//config.warps_mma1_n, 16),
            (config.Br//(Nwarps//config.warps_mma_n), config.D//config.warps_mma_n, 16),
        )
        cost_model_configs.append(cost_model_config)
        print(cost_model_config)
        pprint.pprint(config)

        cycles = ncu_cycles(
            batch, nheads, seqlen_q, seqlen_kv, dim_qk, dim_v,
            "attn", "shared", arch_type, 
            config.Br, config.Bc, config.Nthreads, config.warps_mma1_n, config.warps_mma_n, config.unrollLastIter, config.BlockKSmem, config.BlockKSmem2, config.num_stages_qk, config.num_stages_v
        )
        print(cycles)


