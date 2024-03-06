#include <torch/extension.h>
#include <ATen/ATen.h>
using at::Tensor;

void launch_min_area_rect(const Tensor pointsets,
                                Tensor polygons
);