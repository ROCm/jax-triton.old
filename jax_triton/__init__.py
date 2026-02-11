# Copyright 2025 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for JAX-Triton integrations."""

__all__ = [
  "utils",
  "triton_call",
  "cdiv",
  "next_power_of_2",
  "strides_from_shape",
  "__version__",
  "__version_info__",
]

from jax._src.lib import gpu_triton
from jax_triton import utils
from jax_triton.triton_lib import triton_call
from jax.experimental.pallas import cdiv
from jax.experimental.pallas import next_power_of_2
from jax.experimental.pallas import strides_from_shape
from jax_triton.version import __version__
from jax_triton.version import __version_info__

try:
  get_compute_capability = gpu_triton.get_compute_capability
  get_serialized_metadata = gpu_triton.get_serialized_metadata
except AttributeError:
  raise ImportError(
    "jax-triton requires JAX to be installed with GPU support. The "
    "installation page on the JAX documentation website includes "
    "instructions for installing a supported version:\n"
    "https://jax.readthedocs.io/en/latest/installation.html"
  )
else:
  del gpu_triton  # Not part of the API.

###################### triton.backends.amd.compiler.HIPBackend patch begin ##############
# this is patch fixing a bomb planted into Triton's AMD-specific compilation path by
# https://github.com/triton-lang/triton/commit/37ff43c5efd6e1b84c00a599ba070a501181e832#diff-33c9a103282c05c9d9d213b94450ae7481b6db8c3c6d810f54f175b4735a3c72
# In short: there's an unconditional and totally unnecessary "import torch" directive crashing
# the code when torch isn't installed.
#
# Remove the patch once triton wheel package version is pinned to >= triton version with the fix.
try:
  from triton.backends.amd.compiler import HIPBackend
  # if we're thrown at above, that's fine, Triton not available and it's handled later

  # now testing if the original implementation fails:
  try:
    # passing an int should cause the exception
    HIPBackend.is_within_2gb(1)
    # if we're here, either the torch is installed and it's not a problem, or the code was fixed
    # and it's not a problem anymore too
  except ImportError:
    # redefining poisoned implementation
    def fixed_is_within_2gb(arg):
      # leaving the check as theoretically a user could update package discovery paths and the
      # module becomes importable.
      HAS_TORCH = False
      try:
        import torch  # pytype: disable=import-error

        HAS_TORCH = True
      except ImportError:
        pass

      MAX_INT_32 = 2**31 - 1
      if hasattr(arg, "ptr_range"):
        return arg.ptr_range() <= MAX_INT_32
      if (
        HAS_TORCH and isinstance(arg, torch.Tensor) and hasattr(arg, "untyped_storage")
      ):
        return arg.untyped_storage().size() <= MAX_INT_32
      return False

    HIPBackend.is_within_2gb = fixed_is_within_2gb

except ImportError:
  pass
###################### triton.backends.amd.compiler.HIPBackend patch end ##############
