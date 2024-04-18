import torch
import pprint
from autotuner.runtime import Runtime
from arch import A100, RTX4090
from utils.ref_op import attention_ref, retnet_ref
from utils.misc import is_close_my

from autotuner.configs import AttnConfig, RetConfig, RetBwdConfig

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
    },
    },

]

def test_attn(test_dict):
    arch = test_dict["arch_type"]
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
    
    cc_dict = test_dict["configs"]
    for cc in cc_dict["register"]:
        cc.set_fuse_type("register")
        Runtime(arch, cc,tmp_dir=f"../../tmp/{operation}").apply(torch_array)

        o_ref = ref_func()
        isclose = is_close_my(o, o_ref, rtol=1e-3, atol=1e-3)
        if not isclose:
            print("FAILED CONFIG:")
            pprint.pprint(cc)
    for cc in cc_dict["shared"]:
        cc.set_fuse_type("shared")
        Runtime(arch, cc,tmp_dir=f"../../tmp/{operation}").apply(torch_array)

        o_ref = ref_func()
        isclose = is_close_my(o, o_ref, rtol=1e-3, atol=1e-3)
        if not isclose:
            print("FAILED CONFIG:")
            pprint.pprint(cc)
    

    for cc in bwd_cc_list:
        Runtime(arch, cc, tmp_dir=f"../../tmp/{operation}_bwd").apply(bwd_array)
        o_ref.backward(do)
        dq_ref = q.grad
        dk_ref = k.grad
        dv_ref = v.grad
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
    """
    b = 4
    h = 4
    seq_q = 2048
    seq_kv = 2048
    Kd = 256
    D = 128
    q = torch.randn([b, h, seq_q, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
    k = torch.randn([b, h, seq_kv, Kd], device="cuda:0", dtype=torch.float16, requires_grad=True)
    v = torch.randn([b, h, seq_kv, D], device="cuda:0", dtype=torch.float16, requires_grad=True)
    mask = torch.randn([h, seq_q, seq_kv], device="cuda:0", dtype=torch.float16)
    r = torch.zeros([b, h, seq_q], device="cuda:0", dtype=torch.float32)
    o = torch.zeros([b, h, seq_q, D], device="cuda:0", dtype=torch.float16)
    """


    # RTX 4090--------------------------------
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
