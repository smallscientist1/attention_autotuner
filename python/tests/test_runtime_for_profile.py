import torch
import sys
sys.path.append("../")
from autotuner.runtime import Runtime
import arch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='')
    parser.add_argument('--seqlen_q', type=int, default=2048, help='')
    parser.add_argument('--seqlen_kv', type=int, default=2048, help='')
    parser.add_argument('--nheads', type=int, default=8, help='')
    parser.add_argument('--dim_qk', type=int, default=256, help='')
    parser.add_argument('--dim_v', type=int, default=256, help='')
    parser.add_argument('--operation', type=str, default='attn', help='[retnet, attn]')
    parser.add_argument('--fuse_type', type=str, default="none", help="[register, shared, none]")
    parser.add_argument('--arch', type=str, default="A100", help="[A100, RTX4090]")
    
    parser.add_argument("--Br", type=int)
    parser.add_argument("--Bc", type=int)
    parser.add_argument("--Nthreads", type=int)
    parser.add_argument("--warp_mma1_n", type=int)
    parser.add_argument("--warp_mma_n", type=int)
    parser.add_argument("--unrollLastIter", type=bool)
    parser.add_argument("--BlockKSmem", type=int)
    parser.add_argument("--BlockKSmem2", type=int)
    parser.add_argument("--num_stages_qk", type=int)
    parser.add_argument("--num_stages_v", type=int)

    return parser.parse_args()

if __name__ == "__main__":
    torch.manual_seed(0)
    args = parse_args()
    b = args.batch
    h = args.nheads
    seq_q = args.seqlen_q
    seq_kv = args.seqlen_kv
    Kd = args.dim_qk
    D = args.dim_v
    q = torch.randn([b, h, seq_q, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
    k = torch.randn([b, h, seq_kv, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
    v = torch.randn([b, h, seq_kv, D], device="cuda:0", dtype=torch.float16, requires_grad=True)
    mask = torch.randn([h, seq_q, seq_kv], device="cuda:0", dtype=torch.float16)
    r = torch.zeros([b, h, seq_q], device="cuda:0", dtype=torch.float32)
    o = torch.zeros([b, h, seq_q, D], device="cuda:0", dtype=torch.float16)

    from autotuner.configs import AttnConfig, RetConfig
    if args.operation == 'attn':
        torch_array = [q, k, v, o]
        # cc = AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=2, Nthreads=256)
        cc = AttnConfig(Br=args.Br, Bc=args.Bc, Kd=Kd, D=D, BlockKSmem=args.BlockKSmem, BlockKSmem2=args.BlockKSmem2, num_stages_qk=args.num_stages_qk, num_stages_v=args.num_stages_v, Nthreads=args.Nthreads, unrollLastIter=args.unrollLastIter,warps_mma1_n=args.warp_mma1_n, warps_mma_n=args.warp_mma_n)
    elif args.operation == 'retnet':
        torch_array = [q, k, v, mask, o, r]
        # cc = RetConfig(Br=128, Bc = 128, Kd = 256, D = 256, BlockKSmem=64, BlockKSmem2=128, num_stages_qk=2, num_stages_mask=1, num_stages_v=1, Nthreads=256, unrollLastIter=0)
        cc = RetConfig(Br=args.Br, Bc=args.Bc, Kd=Kd, D=D, BlockKSmem=args.BlockKSmem, BlockKSmem2=args.BlockKSmem2, num_stages_qk=args.num_stages_qk, num_stages_mask=1, num_stages_v=args.num_stages_v, Nthreads=args.Nthreads, unrollLastIter=args.unrollLastIter,warps_mma1_n=args.warp_mma1_n, warps_mma_n=args.warp_mma_n)
    else:
        cc = None
    
    if args.fuse_type == "register":
        cc.set_fuse_type("register")
    elif args.fuse_type == "shared":
        cc.set_fuse_type("shared")

    Runtime(arch.__getattribute__(args.arch)(), cc,tmp_dir="../../tmp/"+cc.operation).apply(torch_array)

