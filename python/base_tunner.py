import ctypes
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile
import subprocess
import importlib.util

import ctypes
import torch
from config import BaseConfig, supported_configs

import pprint
import json



        
class CompileResult:
    def __init__(self, config: BaseConfig, lib_name: str) -> None:
        self.config = config
        self.lib_name = lib_name

def _create_code_for_profiling(config):
    if config.fuse_type == "register":
        profile_code_path = os.path.join(config.template_dir , "regfuse", "profile_code.py")
    elif config.fuse_type == "shared":
        profile_code_path = os.path.join(config.template_dir , "smemfuse", "profile_code.py")
    spec = importlib.util.spec_from_file_location("ProfileCode", profile_code_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    # from template.flash_kernels.retnet.regfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads)
    # from template.flash_kernels.retnet.smemfuse.profile_code import profile_code
    # return profile_code.format(Br=config.Br, Bc=config.Bc, Kd=config.Kd, D=config.D, unrollLastIter=int(config.unrollLastIter), BlockKSmem=config.BlockKSmem, num_stages_qk=config.num_stages_qk, num_stages_mask=config.num_stages_mask, BlockKSmem2=config.BlockKSmem2, num_stages_v=config.num_stages_v, Nthreads=config.Nthreads, warps_mma1_n=config.warps_mma1_n, warps_mma_n=config.warps_mma_n)
    return foo.profile_code.format_map(config.__dict__)

def _compile(config, arch, temp_dir:str, timeout: float = None):
    ## compile
    profiling_code = _create_code_for_profiling(config)
    src = tempfile.NamedTemporaryFile(mode="w",suffix=".cu", delete=True, dir=temp_dir)
    lib_name = src.name.replace(".cu", ".so")
    compute_version = arch.compute_capability
    cutlass_dir = os.path.join(os.path.dirname(__file__), "../third_party/cutlass/include")
    csrc_dir = os.path.join(os.path.dirname(__file__), "../csrc")    
    if config.fuse_type == "register":
        template_dir = os.path.join(config.template_dir , "regfuse/")
    elif config.fuse_type == "shared":
        template_dir = os.path.join(config.template_dir , "smemfuse/")
    command = ["nvcc","-std=c++17","-O3","--use_fast_math","--expt-relaxed-constexpr","--disable-warnings", "--compiler-options", "'-fPIC'", "--shared", src.name, "-lcuda",
            f"-gencode=arch=compute_{compute_version},code=sm_{compute_version}",
            f"-I{cutlass_dir}",f"-I{template_dir}",f"-I{csrc_dir}", "-o", lib_name]
    src.write(profiling_code)
    src.flush()
    try:
        ret = subprocess.run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if ret.returncode != 0:
        return None
    return CompileResult(config,lib_name)

class BaseTunner:
    def __init__(self, arch, torch_array: list, op_name):
        self.arch = arch
        self.torch_array = torch_array
        self.Br_list = [32, 64, 128, 256]
        self.Bc_list = [32, 64, 128, 256]

        self.op_name = op_name
        self.cache_path = os.path.join(os.path.dirname(__file__), "../cache/")
        self.problem_key = {
            "dim_qk": torch_array[0].shape[-1],
            "dim_v": torch_array[2].shape[-1]
        }

    def compile(self, configs:list, temp_dir:str, timeout: float = None):
        libs = []
        for config in configs:
            lib = _compile(config, self.arch, temp_dir, timeout)
            libs.append(lib)
        return libs
    
    def compile_parallel(self, configs:list, temp_dir:str, timeout: float = None):
        ## compile
        arch = self.arch
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            libs = executor.map(_compile, configs,[arch for _ in configs],[temp_dir for _ in configs],[timeout for _ in configs])
        return list(libs)
    
    def profile(self, libname, device="cuda:0") -> float:
        torch_array = self.torch_array
        batch_size = torch_array[0].shape[0]
        nheads = torch_array[0].shape[1]
        seqlen_q = torch_array[0].shape[2]
        seqlen_k = torch_array[1].shape[2]
        lib = ctypes.CDLL(libname)
        lib.profile.restype = ctypes.c_float
        torch.cuda.set_device(device)
        latency = lib.profile(*[ctypes.c_void_p(arr.data_ptr()) for arr in self.torch_array], ctypes.c_int(batch_size), ctypes.c_int(nheads), ctypes.c_int(seqlen_q), ctypes.c_int(seqlen_k))
        if latency < 0:
            latency = 1e8
        # remove lib
        # subprocess.run(["rm", libname], check=True)
        return latency
    
    def tune(self, log_path="../logs/"):
        dim_qk = self.problem_key["dim_qk"]
        dim_v = self.problem_key["dim_v"]

        best_config = self.check_cache()
        if best_config is not None:
            # print("Best config found in cache: ")
            # pprint.pprint(best_config)
            return best_config

        configs = []
        for Br in self.Br_list:
            for Bc in self.Bc_list:
                cur_configs = self.generate_configs(Br,Bc,dim_qk,dim_v)
                for cur_config in cur_configs:
                    if cur_config.fuse_type=="register" and self.validate_register_fuse(cur_config):
                        configs.append(cur_config)
                    elif cur_config.fuse_type=="shared" and self.validate_shared_fuse(cur_config):
                        configs.append(cur_config)

        # print configs
        print("Configs to be tuned: ")
        for config in configs:
            # print(config)
            pprint.pprint(config)

        # create temp dir
        src_dir = tempfile.TemporaryDirectory()

        # cresults = self.compile(configs,src_dir.name,timeout=1200)
        cresults = self.compile_parallel(configs,src_dir.name,timeout=120)
        profile_dict = {}
        latency = 1e8
        best_config = None
        for cresult in cresults:
            if cresult.lib_name is None:
                continue
            lib_latency = self.profile(cresult.lib_name)
            if lib_latency == 1e8:
                # print(cresult.config)
                pprint.pprint(cresult.config)
                print("profile runtime error")
            if lib_latency < latency:
                latency = lib_latency
                best_config = cresult.config
            profile_dict[cresult.config] = lib_latency

        print("##########################################################")
        print("Operation type: ", best_config.operation)
        print("Best config: ")# , best_config)
        pprint.pprint(best_config)
        print("Latency: ", latency)

        file_name = "profile_result_{}_{}_{}.txt".format(best_config.operation,dim_qk, dim_v)
        os.makedirs(log_path,exist_ok=True)
        with open(os.path.join(log_path,file_name),"w") as f:
            for config in profile_dict:
                f.write(repr(config)+"\n")
                f.write(str(profile_dict[config])+"\n")
            f.write("\n")
            f.write("best config: \n")
            f.write(repr(best_config)+"\n")
            f.write(str(latency)+"\n")

        cache_path = self.cache_path
        os.makedirs(cache_path,exist_ok=True)
        with open(os.path.join(cache_path,"best_config_{}_{}_{}.json".format(self.op_name,dim_qk, dim_v)),"w") as f:
            json.dump(best_config.__dict__,f)

        return best_config
    
    def check_cache(self):
        cache_path = self.cache_path
        op_name = self.op_name
        dim_qk = self.problem_key["dim_qk"]
        dim_v = self.problem_key["dim_v"]
        if os.path.exists(os.path.join(cache_path, "best_config_{}_{}_{}.json".format(op_name,dim_qk, dim_v))):
            with open(os.path.join(cache_path,"best_config_{}_{}_{}.json".format(op_name,dim_qk, dim_v)),"r") as f:
                best_config_dict = json.load(f)
            best_config = BaseConfig.from_dict(best_config_dict)
            return best_config
        
        return None
            
        
    def validate_shared_fuse(self, config):
        return False
    def validate_register_fuse(self, config):
        return False
    def generate_configs(self,Br:int,Bc:int,dim_qk:int,dim_v:int):
        configs = []
        return configs

