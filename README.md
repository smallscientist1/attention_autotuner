# attention_autotuner
an autotuner for attention

## build
- clone the repo and submodule
```
git clone --recursive https://github.com/smallscientist1/attention_autotuner.git
```
- build the benchmark on nvidia Ampere GPU(e.g. A100)
```
cd benchmarks
mkdir build
cd build
cmake -DPROJECT_CUDA_ARCH="80" ..
```

## TODO
- chunkwise retnet
- cost model
- autotuner(more general policy for retnet)
- elementwise op
- attention backward
- retnet performance issue(added load q once)
- causal config
- retnet parallel scan version seqlen_q != seqlen_kv
- retnet parallel scan template
