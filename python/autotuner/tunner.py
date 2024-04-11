
import ctypes
import os
import torch

from .base_tunner import CompileResult, BaseTunner
from .configs import RetConfig, AttnConfig
 


class RetTunner(BaseTunner):
    def __init__(self, arch, torch_array: list):
        super().__init__(arch, torch_array, "retnet")
    
    # validate shared fuse
    def validate_shared_fuse(self, config):
        # check shared memory
        smem_size_qk = config.num_stages_qk * config.Br * config.BlockKSmem * 2 + config.num_stages_qk * config.Bc * config.BlockKSmem * 2
        smem_size_v = config.num_stages_v * config.BlockKSmem2 * config.D * 2
        smem_size_mask = config.num_stages_mask * config.Br * config.Bc * 2
        smem_size_accs = config.Br * config.Bc *2 + config.Br * config.Bc * 4 + 2 * config.Br * 4
        smem_out = config.Br * config.D * 2
        smem_size = smem_size_qk + smem_size_v + smem_size_mask + smem_size_accs
        smem_size = max(smem_size, smem_out)
        if smem_size > self.arch.smem_cap:
            return False

        # check register
        Br = config.Br
        Bc = config.Bc
        D = config.D
        Nthreads = config.Nthreads
        reg_used_accum = (Br * D * 4)/(Nthreads * 4)
        reg_used = reg_used_accum
        if reg_used > min(self.arch.register_per_thread, self.arch.reg_cap/Nthreads):
            return False

        return True

    def validate_register_fuse(self, config):
        Br = config.Br
        Bc = config.Bc
        D = config.D
        Nthreads = config.Nthreads
        mmam, mman, mmak = self.arch.cutlass_mma
        belem_per_thread = mman*mmak/self.arch.warp_size
    
        # check tile size
        if Br % (mmam*Nthreads/self.arch.warp_size) != 0:
            return False
        # check shared memory
        smem_size_qk = config.num_stages_qk * config.Br * config.BlockKSmem * 2 + config.num_stages_qk * config.Bc * config.BlockKSmem * 2
        smem_size_v = config.num_stages_v * config.BlockKSmem2 * config.D * 2
        smem_size_mask = config.num_stages_mask * config.Br * config.Bc * 2
        smem_out = config.Br * config.D * 2
        smem_size = smem_size_qk + smem_size_v + smem_size_mask
        smem_size = max(smem_size, smem_out)
        if smem_size > self.arch.smem_cap:
            return False
        # check register
        reg_used_accum = (Br * D * 4 + Br*Bc*4)/(Nthreads * 4)
        reg_used_matmul2 = (Br * D * 4 + Br*Bc*2)/(Nthreads * 4) + (D/(mman*1) * belem_per_thread*2) / 4
        reg_used_matmul1 = (Br * D * 4 + Br * Bc * 4)/(Nthreads * 4) + (Bc/(mman*1) * belem_per_thread*2) / 4
        reg_used = reg_used_accum # max(reg_used_accum, reg_used_matmul2, reg_used_matmul1)
        if reg_used > min(self.arch.register_per_thread, self.arch.reg_cap/Nthreads):
            return False
        return True
        
    def generate_configs(self,Br:int,Bc:int,dim_qk:int,dim_v:int):
        configs = []
        for BlockKSmem,num_stages_qk in [(dim_qk,1),(int(dim_qk/4),2)]:
            if BlockKSmem % 32 != 0:
                continue
            for BlockKSmem2,num_stages_v in [(Bc,1),(int(Bc/4),2)]:
                if BlockKSmem2 % 32 != 0:
                    continue
                # TODO: more general
                Nthreads = 128 if Br==32 and Bc==32 else 256
                num_stages_mask = 1
                for unrollLastIter in [True, False]:
                    config1 = RetConfig(Br,Bc,dim_qk,dim_v,BlockKSmem,BlockKSmem2,num_stages_qk,num_stages_mask,num_stages_v,Nthreads,unrollLastIter)
                    config1.set_fuse_type("register")
                    configs.append(config1)
                    config2 = RetConfig(Br,Bc,dim_qk,dim_v,BlockKSmem,BlockKSmem2,num_stages_qk,num_stages_mask,num_stages_v,Nthreads,unrollLastIter)
                    config2.set_fuse_type("shared")
                    configs.append(config2)
        return configs

class AttnTunner(BaseTunner):
    def __init__(self, arch, torch_array: list):
        super().__init__(arch, torch_array, "attention")

    # validate shared fuse
    def validate_shared_fuse(self, config):
        # check shared memory
        smem_size_qk = config.num_stages_qk * config.Br * config.BlockKSmem * 2 + config.num_stages_qk * config.Bc * config.BlockKSmem * 2
        smem_size_v = config.num_stages_v * config.BlockKSmem2 * config.D * 2
        smem_size_accs = config.Br * config.Bc *2 + config.Br * config.Bc * 4 + 3 * config.Br * 4
        smem_out = config.Br * config.D * 2
        smem_size = smem_size_qk + smem_size_v + smem_size_accs
        smem_size = max(smem_size, smem_out)
        if smem_size > self.arch.smem_cap:
            return False
        
        # check register
        Br = config.Br
        Bc = config.Bc
        D = config.D
        Nthreads = config.Nthreads
        reg_used_accum = (Br * D * 4)/(Nthreads * 4)
        reg_used = reg_used_accum
        if reg_used > min(self.arch.register_per_thread, self.arch.reg_cap/Nthreads):
            return False

        return True
    
    def validate_register_fuse(self, config):
        Br = config.Br
        Bc = config.Bc
        D = config.D
        Nthreads = config.Nthreads
        mmam, mman, mmak = self.arch.cutlass_mma
        belem_per_thread = mman*mmak/self.arch.warp_size
    
        # check tile size
        if Br % (mmam*Nthreads/self.arch.warp_size) != 0:
            return False
        # check shared memory
        smem_size_qk = config.num_stages_qk * config.Br * config.BlockKSmem * 2 + config.num_stages_qk * config.Bc * config.BlockKSmem * 2
        smem_size_v = config.num_stages_v * config.BlockKSmem2 * config.D * 2
        smem_out = config.Br * config.D * 2
        smem_size = smem_size_qk + smem_size_v
        smem_size = max(smem_size, smem_out)
        if smem_size > self.arch.smem_cap:
            return False
        # check register
        reg_used_accum = (Br * D * 4 + Br*Bc*4)/(Nthreads * 4)
        reg_used_matmul2 = (Br * D * 4 + Br*Bc*2)/(Nthreads * 4) + (D/(mman*1) * belem_per_thread*2) / 4
        reg_used_matmul1 = (Br * D * 4 + Br * Bc * 4)/(Nthreads * 4) + (Bc/(mman*1) * belem_per_thread*2) / 4
        reg_used = reg_used_accum # max(reg_used_accum, reg_used_matmul2, reg_used_matmul1)
        if reg_used > min(self.arch.register_per_thread, self.arch.reg_cap/Nthreads):
            return False
        return True
    
    def generate_configs(self,Br:int,Bc:int,dim_qk:int,dim_v:int):
        configs = []
        for BlockKSmem,num_stages_qk in [(dim_qk,1),(64 if dim_qk/2 > 64 else 32 if dim_qk/2 > 32 else 16, 2)]:# dim_qk: load q once; or stage=2 pipelined in matmul1 self(64 for dim_qk large or 32 for dim_qk 128 here)
                                                                            # dim_qk: 64的倍数
            if BlockKSmem % 32 != 0:
                continue
            for BlockKSmem2,num_stages_v in [(Bc,1),(int(Bc/4),2)]:
                if BlockKSmem2 % 32 != 0:
                    continue
                # TODO: more general
                for Nthreads in [128, 256]:
                # Nthreads = 128 if Br==32 and Bc==32 else 256
                    if Br==32 and Bc==32 and Nthreads==256: # matmul1
                        continue
                    if BlockKSmem==32 and BlockKSmem2==32 and Nthreads==256: # matmul2(BlockKSmem=32->SmemKAtom=32-> load v: BlockKSmem2>=Nthreads/(SmemKatom/8))
                        continue
                    for unrollLastIter in [True, False]:
                        config1 = AttnConfig(Br,Bc,dim_qk,dim_v,BlockKSmem,BlockKSmem2,num_stages_qk,num_stages_v,Nthreads,unrollLastIter)
                        config1.set_fuse_type("register")
                        configs.append(config1)
                        config2 = AttnConfig(Br,Bc,dim_qk,dim_v,BlockKSmem,BlockKSmem2,num_stages_qk,num_stages_v,Nthreads,unrollLastIter)
                        config2.set_fuse_type("shared")
                        configs.append(config2)
        return configs

