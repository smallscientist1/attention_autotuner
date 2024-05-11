import torch
import pprint
from autotuner.runtime import Runtime
from arch import A100, RTX4090
from utils.ref_op import attention_ref, retnet_ref
from utils.misc import is_close_my

from autotuner.configs import AttnConfig, RetConfig, RetBwdConfig

device_type = RTX4090() if torch.cuda.get_device_capability(0) == (8,9) else A100() if torch.cuda.get_device_capability(0) == (8,0) else None

test_dict_list = [
    {
    "arch_type": A100(),
    "operation": "attn",
    "problem_size": (4,8, 2048, 2048, 256, 256),
    "configs":{
        "register": [
            AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=2, Nthreads=256),
        ],
        "shared": [
            AttnConfig(Br=64, Bc=64, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=64, num_stages_v=1, Nthreads=256),
            AttnConfig(Br=128, Bc=32, Kd=256, D=256, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=32, num_stages_v=1, Nthreads=256),
            AttnConfig(Br=64, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, Nthreads=256),
        ]
    }
    },
    {
    "arch_type": A100(),
    "operation": "attn",
    "problem_size": (4,8, 2048, 2048, 256, 128),
    "configs":{
        "register": [
            
        ],
        "shared": [
            AttnConfig(Br=64, Bc=64, Kd=256, D=128, unrollLastIter=1, BlockKSmem=256, num_stages_qk=1, BlockKSmem2=64, num_stages_v=1, Nthreads=128),
        ]
    }
    },
    {
    "arch_type": A100(),
    "operation": "attn",
    "problem_size": (4,8, 2048, 2048, 128, 128),
    "configs":{
        "register": [
            AttnConfig(Br=64, Bc=256,Kd=128,D=128, BlockKSmem=128, BlockKSmem2=256, num_stages_qk=1, num_stages_v=1, Nthreads=128, unrollLastIter=1),
        ],
        "shared": [
            AttnConfig(Br=32,Bc=64,Kd=128,D=128,BlockKSmem=128,BlockKSmem2=64,num_stages_qk=1,num_stages_v=1,Nthreads=256,unrollLastIter=1,warps_mma1_n=4,warps_mma_n=4),
            AttnConfig(Br=32,Bc=64,Kd=128,D=128,BlockKSmem=32,BlockKSmem2=64,num_stages_qk=2,num_stages_v=1,Nthreads=128,unrollLastIter=1,warps_mma1_n=2,warps_mma_n=4),
        ]
    }
    },
    {
    "arch_type": A100(),
    "operation": "ret",
    "problem_size": (4,8, 2048, 2048, 256, 256),
    "configs":{
        "register": [
            RetConfig(Br=128, Bc = 128, Kd = 256, D = 256, BlockKSmem=64, BlockKSmem2=128, num_stages_qk=2, num_stages_mask=1, num_stages_v=1, Nthreads=256, unrollLastIter=0),
        ],
        "shared": [
            RetConfig(Br=64, Bc=128, Kd=256, D=256, unrollLastIter=1, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, num_stages_mask=1, Nthreads=256),
        ]
    },
    "bwd_configs":{
        RetBwdConfig(Br=64, Bc=64, Kd=256, D=256, mmawarpsN=2, mmawarpsN_dk=4, mmawarpsN_dv=4, mmawarpsN_dq=4),
        # result not correct
        # RetBwdConfig(Br=64, Bc=64, Kd=256, D=256, mmawarpsN=2, mmawarpsN_dk=4, mmawarpsN_dv=4, mmawarpsN_dq=4,BlockKSmem=64,num_stages_qk=2),
        RetBwdConfig(Br=64, Bc=64, Kd=256, D=256, mmawarpsN=2, mmawarpsN_dk=2, mmawarpsN_dv=2, mmawarpsN_dq=2,Nthreads=128),
    },
    },
    {
    "arch_type": RTX4090(),
    "operation": "attn",
    "problem_size": (4,8, 2048, 2048, 256, 256),
    "configs":{
        "register": [
            AttnConfig(Br=128, Bc=128, Kd=256, D=256, unrollLastIter=0, BlockKSmem=64, num_stages_qk=2, BlockKSmem2=32, num_stages_v=2, Nthreads=256),
        ],
        "shared": [
        
        ],
    }
    }

]

def test_attn(test_dict):
    arch = test_dict["arch_type"]
    if device_type.compute_capability != arch.compute_capability:
        return
    operation = test_dict["operation"]
    b,h, seq_q, seq_kv, Kd, D = test_dict["problem_size"]
    if operation == "attn" or operation == "ret":
        q = torch.randn([b, h, seq_q, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
        k = torch.randn([b, h, seq_kv, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
        v = torch.randn([b, h, seq_kv, D], device="cuda:0", dtype=torch.float16, requires_grad=True)
        r = torch.zeros([b, h, seq_q], device="cuda:0", dtype=torch.float32)
        o = torch.zeros([b, h, seq_q, D], device="cuda:0", dtype=torch.float16)
        if operation == "ret":
            mask = torch.randn([h, seq_q, seq_kv], device="cuda:0", dtype=torch.float16)
    if operation == "attn":
        torch_array = [q, k, v, o]
        ref_func = lambda : attention_ref(q, k, v)
    elif operation == "ret":
        torch_array = [q, k, v, mask, o, r]
        ref_func = lambda : retnet_ref(q, k, v, mask)

    bwd_cc_list = []
    if test_dict.get("bwd_configs"):
        bwd_cc_list = test_dict["bwd_configs"]
        do = torch.randn_like(o)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dq_accum = torch.zeros([b, h, seq_q, Kd], device=dq.device, dtype=torch.float32)
        bwd_array = [q, k, v, mask, do, r, dq, dk, dv, dq_accum]


    print(f"Testing {operation} on {str(arch)} with problem size {test_dict['problem_size']}")
    
    o_ref = ref_func()
    cc_dict = test_dict["configs"]
    for cc in cc_dict["register"]:
        o.zero_()
        cc.set_fuse_type("register")
        Runtime(arch, cc,tmp_dir=f"../../tmp/{operation}").apply(torch_array)

        isclose = is_close_my(o, o_ref, rtol=1e-3, atol=1e-3)
        if not isclose:
            print("FAILED CONFIG:")
            pprint.pprint(cc)
    for cc in cc_dict["shared"]:
        o.zero_()
        cc.set_fuse_type("shared")
        Runtime(arch, cc,tmp_dir=f"../../tmp/{operation}").apply(torch_array)

        isclose = is_close_my(o, o_ref, rtol=1e-3, atol=1e-3)
        if not isclose:
            print("FAILED CONFIG:")
            pprint.pprint(cc)
    
    if len(bwd_cc_list):
        o_ref.backward(do)
        dq_ref = q.grad
        dk_ref = k.grad
        dv_ref = v.grad
    for cc in bwd_cc_list:
        dq.zero_()
        dk.zero_()
        dv.zero_()
        dq_accum.zero_()
        Runtime(arch, cc, tmp_dir=f"../../tmp/{operation}_bwd").apply(bwd_array)
        is1 = is_close_my(dq, dq_ref, rtol=1e-3, atol=1e-3)
        is2 = is_close_my(dk, dk_ref, rtol=1e-3, atol=1e-3)
        is3 = is_close_my(dv, dv_ref, rtol=1e-3, atol=1e-3)
        if not is1 or not is2 or not is3:
            print("FAILED CONFIG:")
            pprint.pprint(cc)


if __name__ == "__main__":
    torch.manual_seed(0)
    for test_dict in test_dict_list:
        test_attn(test_dict)
