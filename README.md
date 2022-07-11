# min-LLM

Minimal code to train a relatively large language model (1-10B parameters).

* Minimal codebase to learn and adapt for your own use cases
* Concise demonstration of tricks to optimally train a larger language model
* Allows exploration of compute optimal models at smaller sizes based on realistic scaling laws

The project was inspired by [megatron](https://github.com/NVIDIA/Megatron-LM) and all sub-variants. This repo can be seen as a condensed variant, where some of the very large scaling tricks are stripped out for the sake of readability/simplicity.

For example, the library does not include Tensor Parallelism/Pipeline Parallelism. If you need to reach those 100B+ parameter models, I suggest looking at [megatron](https://github.com/NVIDIA/Megatron-LM).

## Setup

Make sure you're installing/running on a CUDA supported machine.

To improve performance, we use a few fused kernel layers from Apex (if you're unsure what fused kernels are for, I highly suggest [this](https://horace.io/brrr_intro.html) blogpost).

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Install the rest of the requirements:

```
pip install -r requirements.txt
```

## Train

To train a 1.5B parameter model (based on the megatron paper sizes) on 8 GPUs:

```
deepspeed --num_gpus 8 train.py --batch_size_per_gpu 36
```

## References

Code: 

Papers + notes: