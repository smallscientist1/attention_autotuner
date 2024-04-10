template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, bool unrollLastIter_=true>
class ImplementShapeBase{
public:
    constexpr static int Br = Br_;
    constexpr static int Bc = Bc_;
    constexpr static int Kd = Kd_;
    constexpr static int D = D_;
    constexpr static int Nthreads = Nthreads_;
    constexpr static int BlockKSmem = BlockKSmem_;
    constexpr static int num_stages_qk = num_stages_qk_;
    constexpr static bool load_q_once = (BlockKSmem == Kd);
    constexpr static int SmemKAtom = BlockKSmem % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzle = SmemKAtom == 32 ? 2 : 3;
    constexpr static bool unrollLastIter = unrollLastIter_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, int BlockKSmem2_=Bc_, int num_stages_v_=1, bool unrollLastIter_=true>
class ImplementShapeAttnRegFwd: public ImplementShapeBase<Br_, Bc_, Kd_, D_, Nthreads_, BlockKSmem_, num_stages_qk_, unrollLastIter_>{
public:
    constexpr static int BlockKSmem2 = BlockKSmem2_;
    constexpr static int num_stages_v = num_stages_v_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
/*smem_fuse only*/ int warps_mma1_N_, int warps_mma_N_, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, int BlockKSmem2_=Bc_, int num_stages_v_=1, bool unrollLastIter_=true>
class ImplementShapeAttnSharedFwd: public ImplementShapeBase<Br_, Bc_, Kd_, D_, Nthreads_, BlockKSmem_, num_stages_qk_, unrollLastIter_>{
public:
    constexpr static int BlockKSmem2 = BlockKSmem2_;
    constexpr static int num_stages_v = num_stages_v_;

    /*smem_fuse*/
    constexpr static int SmemKAtomV = D_ % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
    constexpr static int SmemKAtomP = Bc_ % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleP = SmemKAtomP == 32 ? 2 : 3;
    constexpr static int SmemKAtomPf16 = 64;
    constexpr static int kSwizzlePf16 = SmemKAtomPf16 == 32 ? 2 : 3;
    constexpr static int warps_mma1_N = warps_mma1_N_;
    constexpr static int warps_mma_N = warps_mma_N_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, int BlockKSmem2_=Bc_, int num_stages_v_=1, bool unrollLastIter_=true,
/*retnet*/int num_stages_mask_ = 1>
class ImplementShapeRetRegFwd: public ImplementShapeAttnRegFwd<Br_, Bc_, Kd_, D_, Nthreads_, BlockKSmem_, num_stages_qk_, BlockKSmem2_, num_stages_v_, unrollLastIter_>{
public:

    /*retnet*/
    constexpr static int SmemKAtomMask = Bc_ % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
    constexpr static int num_stages_mask = num_stages_mask_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
/*smem_fuse only*/ int warps_mma1_N_, int warps_mma_N_, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, int BlockKSmem2_=Bc_, int num_stages_v_=1, bool unrollLastIter_=true,
/*retnet*/int num_stages_mask_ = 1>
class ImplementShapeRetSharedFwd: public ImplementShapeAttnSharedFwd<Br_, Bc_, Kd_, D_, Nthreads_, warps_mma1_N_, warps_mma_N_, BlockKSmem_, num_stages_qk_, BlockKSmem2_, num_stages_v_, unrollLastIter_>{
public:
    /*retnet*/
    constexpr static int SmemKAtomMask = Bc_ % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
    constexpr static int num_stages_mask = num_stages_mask_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
/*smem_fuse only*/ int warps_mma1_N_ = 1, int warps_mma_N_ = 1, 
int BlockKSmem_=Kd_, int num_stages_qk_=1, int BlockKSmem2_=Bc_, int num_stages_v_=1, bool unrollLastIter_=true,
/*retnet*/int num_stages_mask_ = 1>
class ImplementShape{
public:
    constexpr static int Br = Br_;
    constexpr static int Bc = Bc_;
    constexpr static int Kd = Kd_;
    constexpr static int D = D_;
    constexpr static int Nthreads = Nthreads_;
    constexpr static int BlockKSmem = BlockKSmem_;
    constexpr static int num_stages_qk = num_stages_qk_;
    constexpr static bool load_q_once = (BlockKSmem == Kd);
    constexpr static int BlockKSmem2 = BlockKSmem2_;
    constexpr static int num_stages_v = num_stages_v_;
    constexpr static int SmemKAtom = BlockKSmem % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzle = SmemKAtom == 32 ? 2 : 3;
    constexpr static bool unrollLastIter = unrollLastIter_;

    /*smem_fuse*/
    constexpr static int SmemKAtomV = D % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
    constexpr static int SmemKAtomP = Bc % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleP = SmemKAtomP == 32 ? 2 : 3;
    constexpr static int SmemKAtomPf16 = 64;
    constexpr static int kSwizzlePf16 = SmemKAtomPf16 == 32 ? 2 : 3;
    constexpr static int warps_mma1_N = warps_mma1_N_;
    constexpr static int warps_mma_N = warps_mma_N_;

    /*retnet*/
    constexpr static int SmemKAtomMask = Bc % 64 == 0 ? 64 : 32;
    constexpr static int kSwizzleMask = SmemKAtomMask == 32 ? 2 : 3;
    constexpr static int num_stages_mask = num_stages_mask_;
};

template<int Br_, int Bc_,int Kd_, int D_, int Nthreads_, 
/*retnet_bwd*/ int mmawarpsN_, int mmawarpsN_dv_, int mmawarpsN_dk_, int mmawarpsN_dq_,
int BlockKSmem_=Kd_, int num_stages_qk_=1, bool unrollLastIter_=true,
/*retnet*/int num_stages_mask_ = 1, 
/*retnet_bwd*/int num_stages_dv_ = 1, int num_stages_ds_ = 1, int num_stages_dq_=1>
class ImplementShapeBwd: public ImplementShape<Br_, Bc_, Kd_, D_, Nthreads_,1,1, BlockKSmem_, num_stages_qk_,1,1, unrollLastIter_, num_stages_mask_>{
public:
    constexpr static int mmawarpsN = mmawarpsN_;
    constexpr static int mmawarpsN_dv = mmawarpsN_dv_;
    constexpr static int mmawarpsN_dk = mmawarpsN_dk_;
    constexpr static int mmawarpsN_dq = mmawarpsN_dq_;
    constexpr static int num_stages_dv = num_stages_dv_;
    constexpr static int num_stages_ds = num_stages_ds_;
    constexpr static int num_stages_dq = num_stages_dq_;

    constexpr static int SmemKAtomO = 64;
    constexpr static int kSwizzleO = SmemKAtomO == 32 ? 2 : 3;
    constexpr static int SmemKAtomV = 64;
    constexpr static int kSwizzleV = SmemKAtomV == 32 ? 2 : 3;
    constexpr static int SmemKAtom = 64;
    constexpr static int kSwizzle = SmemKAtom == 32 ? 2 : 3;
    constexpr static int SmemKAtomS = 64;
    constexpr static int kSwizzleS = SmemKAtomS == 32 ? 2 : 3;
};

class ProblemShape{
public:
    ProblemShape(int batch,int head,int seqlen_q,int seqlen_kv):B(batch),H(head),Seq_q(seqlen_q),Seq_k(seqlen_kv){};

    int B,H,Seq_q,Seq_k;
};

class recurrentShape: public ProblemShape{
public:
    recurrentShape(int batch,int head,int seqlen_q,int seqlen_kv,int dim_qk, int dim_v):ProblemShape(batch,head,seqlen_q,seqlen_kv),dim_qk(dim_qk),dim_v(dim_v),block_dimqk(dim_qk){

    };

    int dim_qk,dim_v,block_dimqk;
};

