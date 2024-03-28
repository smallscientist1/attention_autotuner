# attention_autotuner
an autotuner for attention

## buld
```
cmake -DPROJECT_CUDA_ARCH="80"
```
## TODO
- chunkwise retnet
- cost model
- autotuner
- elementwise op
- attention backward
- retnet performance issue(load q once?)
- causal config
- retnet parallel scan version seqlen_q != seqlen_kv
- retnet parallel scan template
