#include "min_area_rect.h"

void torch_launch_min_area_rect(const torch::Tensor &pointsets,
                                torch::Tensor &polygons) 
{
        launch_min_area_rect(pointsets.data(), polygons.data());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_min_area_rect",
          &torch_launch_min_area_rect,
          "min_area_rect kernel warpper");
}

TORCH_LIBRARY(min_area_rect, m) {
    m.def("torch_launch_min_area_rect", torch_launch_min_area_rect);
}