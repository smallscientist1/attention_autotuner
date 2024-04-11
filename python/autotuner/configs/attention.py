import os
from .base_config import BaseConfig

class AttnConfig(BaseConfig):
    def __init__(self, Br, Bc, Kd, D, BlockKSmem=256, BlockKSmem2=64, num_stages_qk=1, num_stages_v=1, Nthreads=256, unrollLastIter: bool = True) -> None:
        super().__init__(Br, Bc, Kd, D, BlockKSmem, BlockKSmem2, num_stages_qk, num_stages_v, Nthreads, unrollLastIter)
        self.operation = "attn"
        self.template_dir = os.path.join(os.path.dirname(__file__), "../../../csrc/kernels/attention")
