# Installation for ROCm

## 1. install jax-triton with upstream-main (0.3.0)

Clone the repo with:
```bash
$ git clone https://github.com/jax-ml/jax-triton.git
```

and do an editable install with:
```bash
$ cd jax-triton
$ pip install -e .
```

## 2. install compatible JAX for ROCm

`jax-triton-0.3.0` requires `jax==0.6.0`.

Clone the repo and checkout the branch:
```bash
$ git clone https://github.com/ROCm/jax.git
$ cd jax
$ git checkout rocm-jaxlib-v0.6.0
```

Install XLA dependencies for JAX:
```bash
$ git clone https://github.com/ROCm/xla.git
$ cd xla
$ git checkout rocm-jaxlib-v0.6.0
```

Build JAX for ROCm using the provided script:
```bash
$ bash build_jax_rocm.sh
```