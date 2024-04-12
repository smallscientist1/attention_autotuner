from .base_config import BaseConfig
from .attention import AttnConfig
from .retnet import RetConfig
from .retnet_bwd import RetBwdConfig

supported_configs = {
    "attn": AttnConfig,
    "ret": RetConfig,
    "ret_bwd": RetBwdConfig,
}
