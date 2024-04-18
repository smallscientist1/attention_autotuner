# script to test cycles of kernel
# 
import subprocess
import re
import sys

NCU_EXE = "/usr/local/cuda/bin/ncu"
PYTHON_EXE = sys.executable # "/anaconda/envs/mamba/bin/python"

def ncu_cycles(batch:int, heads, seqlen_q, seqlen_kv, dim_qk, dim_v,
               operation, fuse_type, arch, 
               Br, Bc, Nthreads, warp_mma1_n, warp_mma_n, unrollLastIter, BlockKSmem, BlockKSmem2, num_stages_qk, num_stages_v):
# sudo /usr/local/cuda/bin/ncu  --force-overwrite --target-processes application-only --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0  --section SpeedOfLight  --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source yes --check-exit-code yes python test_runtime_for_profile.py --batch 4 --seqlen_q 2048 --seqlen_kv 2048 --nheads 8 --dim_qk 256 --dim_v 256 --operation attn --fuse_type none --arch A100
    command = ["sudo", NCU_EXE, "--force-overwrite", "--target-processes", "application-only", "--replay-mode", "kernel","--kernel-name", "regex:\"fwd|bwd\"" , "--kernel-name-base", "function", "--launch-skip-before-match", "0", "--section", "SpeedOfLight", "--profile-from-start", "1", "--cache-control", "all", "--clock-control", "none", "--apply-rules", "yes", "--import-source", "yes", "--check-exit-code", "yes", PYTHON_EXE, "test_runtime_for_profile.py"]
    command += ["--batch", str(batch), "--seqlen_q", str(seqlen_q), "--seqlen_kv", str(seqlen_kv), "--nheads", str(heads), "--dim_qk", str(dim_qk), "--dim_v", str(dim_v), "--operation", operation, "--fuse_type", fuse_type, "--arch", arch]
    command += ["--Br", str(Br), "--Bc", str(Bc), "--Nthreads", str(Nthreads), "--warp_mma1_n", str(warp_mma1_n), "--warp_mma_n", str(warp_mma_n), "--unrollLastIter", str(unrollLastIter), "--BlockKSmem", str(BlockKSmem), "--BlockKSmem2", str(BlockKSmem2), "--num_stages_qk", str(num_stages_qk), "--num_stages_v", str(num_stages_v)]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("ERR!")
        print(result.stderr)

    # print(result.stdout)    
    # print(result.stderr)

    text = result.stdout
    # num_pattern1 = r"\d+" # A100(123456)
    num_pattern2 = r"\d+(?:,\d{3})*(?:\.\d+)?" # RTX4090(123,456)
    pattern = r"Elapsed Cycles\s+cycle\s+({})".format(num_pattern2)
    match_list = re.findall(pattern, text)
    assert len(match_list) == 1
    cycles = match_list[0]
    cycles = cycles.replace(",", "")
    # print(cycles)
    return int(cycles)

configs_dict = {
    "A100": (
        (
            (2048, 2048, 256, 128),
            (32, 32, 256, 128),
            (16, 16, 16),
            (32, 32, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (32, 64, 256, 128),
            (16, 32, 16),
            (32, 32, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (64, 64, 256, 128),
            (32, 32, 16),
            (32, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (64, 64, 256, 128),
            (32, 32, 16),
            (64, 32, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (64, 64, 256, 128),
            (16, 32, 16),
            (32, 32, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (128, 64, 256, 128),
            (32, 32, 16),
            (32, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (128, 64, 256, 128),
            (16, 64, 16),
            (32, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 128),
            (128, 64, 256, 128),
            (32, 32, 16),
            (128, 16, 16),
            4,8
        ),
        
        (
            (2048, 2048, 256, 256),
            (64, 64, 256, 256),
            (32, 32, 16),
            (64, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 256),
            (128, 32, 256, 256),
            (32, 16, 16),
            (64, 64, 16),
            4,8
        ),
    ),
    "RTX4090": (
        (
            (2048, 2048, 256, 128),
            (64, 32, 256, 128),
            (16, 32, 16),
            (32, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 256),
            (64, 32, 256, 256),
            (16, 16, 16),
            (64, 64, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 384),
            (64, 32, 256, 384),
            (16, 16, 16),
            (64, 48, 16),
            4,8
        ),
        (
            (2048, 2048, 256, 512),
            (64, 32, 256, 512),
            (16, 16, 16),
            (64, 64, 16),
            4,8
        ),
    )
}
import torch
arch = "RTX4090" if torch.cuda.get_device_capability(0) == (8,9) else "A100" if torch.cuda.get_device_capability(0) == (8,0) else None
if __name__ == "__main__":
    configs = configs_dict[arch]
    for config in configs:
        seqlen_q, seqlen_kv, dim_qk, dim_v = config[0]
        Br, Bc, _, _ = config[1]
        warp_mma1_n = int(config[1][1]/config[2][1])
        warp_mma_n = int(dim_v/config[3][1])
        batch,head = config[4], config[5]
        Nthreads = int(config[1][0]/config[2][0] * (config[1][1]/config[2][1]) * 32)
        cycles = ncu_cycles(
            batch, head, seqlen_q, seqlen_kv, dim_qk, dim_v,
            "attn", "shared", arch, 
            Br, Bc, Nthreads, warp_mma1_n, warp_mma_n, 1, dim_qk, Bc, 1, 1
        )
        print(config)
        print(cycles)
