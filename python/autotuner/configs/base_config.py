import os

def find_factors(n:int):
    factors = []
    for i in range(1,n+1):
        if n%i==0:
            factors.append(i)
    return factors

def find_factor_pairs(n:int):
    factor_pairs = []
    for i in range(1,n+1):
        if n%i==0:
            factor_pairs.append((i,n//i))
    return factor_pairs

def find_best_pair(pair_list:list, target_pair:tuple):
    min_cost = 1.
    best_pair = None
    for pair in pair_list:
        cost = abs(pair[0]*target_pair[1]/(pair[1]*target_pair[0]+pair[0]*target_pair[1]) - 0.5)
        # TODO: should not be here, for A100
        if target_pair[1]%(pair[1]*16)!=0 or target_pair[0]%(pair[0]*16)!=0:
            continue
        if cost < min_cost:
            min_cost = cost
            best_pair = pair
    return best_pair

class BaseConfig:
    def __init__(self, Br, Bc, Kd, D, BlockKSmem=256, BlockKSmem2=64, num_stages_qk=1, num_stages_v=1, Nthreads=256, unrollLastIter:bool = True) -> None:
        self.Br = Br
        self.Bc = Bc
        self.Kd = Kd
        self.D = D
        self.BlockKSmem = BlockKSmem
        self.BlockKSmem2 = BlockKSmem2
        self.num_stages_qk = num_stages_qk # [1,2]
        self.num_stages_v = num_stages_v      # [1,2]
        self.Nthreads = Nthreads
        self.unrollLastIter = int(unrollLastIter)
        # choose warps_mma1_n and warps_mma_n
        Nwarps = Nthreads//32
        Nwarps_factor_pairs = find_factor_pairs(Nwarps)
        _, self.warps_mma1_n = find_best_pair(Nwarps_factor_pairs, (Br,Bc))
        _, self.warps_mma_n = find_best_pair(Nwarps_factor_pairs, (Br,D))
        # self.warps_mma1_n = 4 if Br%64!=0 and Nthreads%256==0 else 2
        # self.warps_mma_n = 4

        self.fuse_type = "None"

        self.operation = None # ["attn" ,"ret"]
        self.template_dir = None

    def set_fuse_type(self, fuse_type):
        self.fuse_type = fuse_type

    def __repr__(self) -> str:
        if self.fuse_type == "register":
            return "Config(fuse_type={}, Br={}, Bc={}, BlockKSmem={}, BlockKSmem2={}, num_stages_qk={}, num_stages_v={}, Nthreads={}, unrollLastIter={})".format(self.fuse_type, self.Br, self.Bc, self.BlockKSmem, self.BlockKSmem2, self.num_stages_qk, self.num_stages_v, self.Nthreads, self.unrollLastIter)
        else:
            return "Config(fuse_type={}, Br={}, Bc={}, BlockKSmem={}, BlockKSmem2={}, num_stages_qk={}, num_stages_v={}, Nthreads={}, unrollLastIter={},warps_mma1_n={},warp_mma_n={})".format(self.fuse_type, self.Br, self.Bc, self.BlockKSmem, self.BlockKSmem2, self.num_stages_qk, self.num_stages_v, self.Nthreads, self.unrollLastIter, self.warps_mma1_n, self.warps_mma_n)

    def __str__(self) -> str:
        if self.fuse_type == "register":
            return f"{self.fuse_type}_{self.Br}_{self.Bc}_{self.Kd}_{self.D}_{self.BlockKSmem}_{self.BlockKSmem2}_{self.num_stages_qk}_{self.num_stages_v}_{self.Nthreads}_{self.unrollLastIter}"
        else:
            return f"{self.fuse_type}_{self.Br}_{self.Bc}_{self.Kd}_{self.D}_{self.BlockKSmem}_{self.BlockKSmem2}_{self.num_stages_qk}_{self.num_stages_v}_{self.Nthreads}_{self.unrollLastIter}_{self.warps_mma1_n}_{self.warps_mma_n}"
        
    @classmethod
    def from_dict(cls, dd:dict):
        cc = cls.__new__(cls) # cls: 子类
        cc.__dict__.update(dd)
        return cc

