import os
from .base_config import BaseConfig

class RetBwdConfig(BaseConfig):
    def __init__(self, Br, Bc, Kd, D, mmawarpsN, mmawarpsN_dv, mmawarpsN_dk, mmawarpsN_dq, Nthreads=256, unrollLastIter: bool = True, BlockKSmem=256, num_stages_qk=1, num_stages_mask=1, num_stages_dv=1, num_stages_ds=1, num_stages_dq=1) -> None:
        super().__init__(Br, Bc, Kd, D, BlockKSmem, num_stages_qk, Nthreads, unrollLastIter)
        self.num_stages_mask = num_stages_mask

        self.mmawarpsN = mmawarpsN
        self.mmawarpsN_dv = mmawarpsN_dv
        self.mmawarpsN_dk = mmawarpsN_dk
        self.mmawarpsN_dq = mmawarpsN_dq
        self.num_stages_dv = num_stages_dv
        self.num_stages_ds = num_stages_ds
        self.num_stages_dq = num_stages_dq
        
        self.operation = "ret_bwd"
        self.template_dir = os.path.join(os.path.dirname(__file__), "../../../csrc/kernels/retnet/bwd")

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        return f"{self.operation}_{self.Br}_{self.Bc}_{self.Kd}_{self.D}_{self.mmawarpsN}_{self.mmawarpsN_dv}_{self.mmawarpsN_dk}_{self.mmawarpsN_dq}_{self.Nthreads}_{self.unrollLastIter}_{self.BlockKSmem}_{self.num_stages_qk}_{self.num_stages_mask}_{self.num_stages_dv}_{self.num_stages_ds}_{self.num_stages_dq}"
