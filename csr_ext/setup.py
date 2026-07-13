from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="quant_matmul",
    ext_modules=[
        CppExtension(
            "quant_matmul",
            ["quant_matmul.cpp"],
            extra_compile_args=["-O3", "-march=native", "-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
