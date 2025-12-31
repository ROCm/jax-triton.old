#!/bin/bash
 
MYDIR=$(dirname "$0")
pushd $MYDIR
 
rm -rf dist/*
ROCM_PATH=$(realpath /opt/rocm)
CLANG_PATH=/usr/lib/llvm-18/bin/clang
XLA_PATH=../xla
GPU_TARGET=$(rocminfo 2>/dev/null | grep -o 'gfx[0-9]\+' | head -1)
 
python3 -m pip uninstall jax -y
python3 -m pip uninstall jaxlib -y
python3 -m pip uninstall \
    jax-rocm60-pjrt jax-rocm60-plugin \
    jax-rocm7-pjrt  jax-rocm7-plugin -y
 
python3 ./build/build.py build \
    --wheels=jaxlib,jax-rocm-plugin,jax-rocm-pjrt \
    --use_clang=true --clang_path=$CLANG_PATH \
    --rocm_version=60 --rocm_path=$ROCM_PATH \
    --bazel_options=--override_repository=xla=$XLA_PATH \
    --rocm_amdgpu_targets=$GPU_TARGET && \
python3 setup.py develop --user && python3 -m pip install dist/*.whl
 
popd