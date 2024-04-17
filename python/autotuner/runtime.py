from .configs import BaseConfig
import os
import importlib.util
import tempfile
import subprocess
import ctypes
import torch
from arch import Arch
'''
compile the kernel library and provide the interface
input: Configs
'''

NVCC_EXE = "/usr/local/cuda/bin/nvcc"

def _create_entry_code(config):
    if config.fuse_type == "register":
        entry_code_path = os.path.join(config.template_dir , "regfuse", "profile_code.py")
    elif config.fuse_type == "shared":
        entry_code_path = os.path.join(config.template_dir , "smemfuse", "profile_code.py")
    else: # bwd
        entry_code_path = os.path.join(config.template_dir , "profile_code.py")
    spec = importlib.util.spec_from_file_location("EntryCode", entry_code_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    # from template.flash_kernels.retnet.regfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads)
    # from template.flash_kernels.retnet.smemfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads, warps_mma1_n=config.warps_mma1_n, warps_mma_n=config.warps_mma_n)
    return foo.kernel_code.format_map(config.__dict__)

class Runtime:
    def __init__(self, arch:Arch, config:BaseConfig,tmp_dir = "../tmp", libname=None):
        self.arch = arch
        self.config = config
        self.tmp_dir = tmp_dir
        self.libname = libname
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        if self.libname is None:
            self.libname = self.compile()

    def compile(self, timeout: float = None):
        temp_dir = self.tmp_dir
        config = self.config
        entry_code = _create_entry_code(config)
        filename = str(config) + ".cu"
        lib_name = filename.replace(".cu", ".so")
        if os.path.exists(os.path.join(temp_dir, lib_name)):
            return lib_name
        with open(os.path.join(temp_dir, filename), "w") as f:
            f.write(entry_code)
            f.flush()
        compute_version = self.arch.compute_capability
        cutlass_dir = os.path.join(os.path.dirname(__file__), "../../third_party/cutlass/include")
        csrc_dir = os.path.join(os.path.dirname(__file__), "../../csrc")
        if config.fuse_type == "register":
            template_dir = os.path.join(config.template_dir , "regfuse/")
        elif config.fuse_type == "shared":
            template_dir = os.path.join(config.template_dir , "smemfuse/")
        else: # bwd
            template_dir = config.template_dir
        command = [NVCC_EXE,"-std=c++17","-O3","--use_fast_math","--expt-relaxed-constexpr","--disable-warnings", "--compiler-options", "'-fPIC'", "--shared", os.path.join(temp_dir, filename), "-lcuda",
            f"-gencode=arch=compute_{compute_version},code=sm_{compute_version}",
            f"-I{cutlass_dir}",f"-I{template_dir}",f"-I{csrc_dir}", "-o", os.path.join(temp_dir, lib_name)]
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if ret.returncode != 0:
            return None
        return lib_name
    
    def apply(self, torch_array:list, device="cuda:0"):
        batch_size = torch_array[0].shape[0]
        nheads = torch_array[0].shape[1]
        seqlen_q = torch_array[0].shape[2]
        seqlen_k = torch_array[1].shape[2]
        lib = ctypes.CDLL(os.path.join(self.tmp_dir, self.libname))
        lib.kernel_entry.restype = ctypes.c_int
        torch.cuda.set_device(device)
        ret = lib.kernel_entry(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_array], ctypes.c_int(batch_size), ctypes.c_int(nheads), ctypes.c_int(seqlen_k), ctypes.c_int(seqlen_q))
        

