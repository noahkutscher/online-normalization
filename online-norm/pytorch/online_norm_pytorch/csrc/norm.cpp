/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2019 Cerebras Systems Inc.
 * All rights reserved.
 *
 * pybind norm cpp / cuda implementations
 *
 * Author:  Vitaliy Chiley
 * Contact: {vitaliy, info}@cerebras.net
 */

#include "norm/norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_fwd", &norm_fwd, "norm_fwd");
  m.def("norm_bwd", &norm_bwd, "norm_bwd");
  m.def("layer_scaling_fwd", &layer_scaling_fwd, "layer_scaling_fwd");
  m.def("layer_scaling_bwd", &layer_scaling_bwd, "layer_scaling_bwd");
}
