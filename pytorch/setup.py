from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="min_area_rect",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "min_area_rect",
            ["pytorch/min_area_rect_ops.cpp", "kernel/min_area_rect.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)