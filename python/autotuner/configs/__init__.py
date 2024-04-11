from .base_config import BaseConfig
from .attention import AttnConfig
from .retnet import RetConfig

supported_configs = {
    "attn": AttnConfig,
    "ret": RetConfig
}
