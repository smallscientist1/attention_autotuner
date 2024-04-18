import torch.nn.functional as F
def attention_ref(q,k,v):
    softmax_scale = 0.125
    attn = q @ k.transpose(-1, -2)
    o_ref = F.softmax(attn * softmax_scale, dim=-1) @ v
    return o_ref

def retnet_ref(q,k,v,mask):
    attn = q @ k.transpose(-1, -2)
    qkm = attn * mask
    r_ref = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o_ref = (qkm/r_ref) @ v
    return o_ref