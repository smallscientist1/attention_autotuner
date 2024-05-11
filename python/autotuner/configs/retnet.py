import os
from .base_config import BaseConfig

class RetConfig(BaseConfig):
    def __init__(self, Br, Bc, Kd, D, BlockKSmem=256, BlockKSmem2=64, num_stages_qk=1, num_stages_mask=1, num_stages_v=1, Nthreads=256, unrollLastIter:bool = True, warps_mma1_n = None, warps_mma_n = None) -> None:
        super().__init__(Br, Bc, Kd, D, BlockKSmem, num_stages_qk, Nthreads, unrollLastIter)
        
        self.BlockKSmem2 = BlockKSmem2
        self.num_stages_v = num_stages_v
        self.num_stages_mask = num_stages_mask # [1]

        if warps_mma1_n is not None:
            self.warps_mma1_n = warps_mma1_n
        if warps_mma_n is not None:
            self.warps_mma_n = warps_mma_n
        
        self.operation = "ret"
        self.template_dir = os.path.join(os.path.dirname(__file__), "../../../csrc/kernels/retnet")
    
    def __repr__(self) -> str:
        if self.fuse_type == "register":
            return "Config(fuse_type={}, Br={}, Bc={}, BlockKSmem={}, BlockKSmem2={}, num_stages_qk={}, num_stages_mask={}, num_stages_v={}, Nthreads={}, unrollLastIter={})".format(self.fuse_type, self.Br, self.Bc, self.BlockKSmem, self.BlockKSmem2, self.num_stages_qk, self.num_stages_mask, self.num_stages_v, self.Nthreads, self.unrollLastIter)
        else:
            return "Config(fuse_type={}, Br={}, Bc={}, BlockKSmem={}, BlockKSmem2={}, num_stages_qk={}, num_stages_mask={}, num_stages_v={}, Nthreads={}, unrollLastIter={},warps_mma1_n={},warp_mma_n={})".format(self.fuse_type, self.Br, self.Bc, self.BlockKSmem, self.BlockKSmem2, self.num_stages_qk, self.num_stages_mask, self.num_stages_v, self.Nthreads, self.unrollLastIter, self.warps_mma1_n, self.warps_mma_n)
    
    def __str__(self) -> str:
        if self.fuse_type == "register":
            return f"{self.fuse_type}_{self.Br}_{self.Bc}_{self.Kd}_{self.D}_{self.BlockKSmem}_{self.BlockKSmem2}_{self.num_stages_qk}_{self.num_stages_mask}_{self.num_stages_v}_{self.Nthreads}_{self.unrollLastIter}"
        else:
            return f"{self.fuse_type}_{self.Br}_{self.Bc}_{self.Kd}_{self.D}_{self.BlockKSmem}_{self.BlockKSmem2}_{self.num_stages_qk}_{self.num_stages_mask}_{self.num_stages_v}_{self.Nthreads}_{self.unrollLastIter}_{self.warps_mma1_n}_{self.warps_mma_n}"
