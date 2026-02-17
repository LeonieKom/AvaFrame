from setuptools import setup, Extension
import numpy
import os
import platform

from Cython.Build import cythonize

# Detect platform for compiler flags
is_windows = platform.system() == 'Windows'

# Compiler optimization flags
# Set DISABLE_OPTIMIZATIONS=1 to fall back to safe defaults (backwards compatibility)
disable_optimizations = os.environ.get('DISABLE_OPTIMIZATIONS', '0') == '1'

if disable_optimizations:
    # Safe fallback: no aggressive optimizations
    extra_compile_args = []
    define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    print("WARNING: Compiler optimizations disabled (DISABLE_OPTIMIZATIONS=1)")
else:
    # Production mode: maximum optimization
    # Windows (MSVC): /Ox = maximum optimization, /fp:fast = fast floating point
    # Linux (GCC):    -O3 = maximum optimization, -ffast-math = fast floating point
    if is_windows:
        extra_compile_args = ['/Ox', '/fp:fast']
        define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    else:
        extra_compile_args = ['-O3', '-ffast-math', '-march=native']
        define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# Set CYTHON_TRACE=1 environment variable to enable linetrace for profiling
enable_linetrace = os.environ.get('CYTHON_TRACE', '0') == '1'
if enable_linetrace:
    define_macros.append(('CYTHON_TRACE', '1'))
    print("INFO: Cython line tracing enabled (CYTHON_TRACE=1) - performance will be reduced")

# List all potential Cython file locations
cython_files = [
    'avaframe/com1DFA/DFAfunctionsCython.pyx',
    'avaframe/com1DFA/DFAToolsCython.pyx',
    'avaframe/com1DFA/damCom1DFA.pyx',
]

extensions = [
    Extension(
        name=file.replace('/', '.').replace('.pyx', ''),
        sources=[file],
        include_dirs=[numpy.get_include()],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    for file in cython_files
]

ext_modules = cythonize(
    extensions,
    compiler_directives={
        'language_level': '3',
        'linetrace': enable_linetrace,  # Only enable for profiling (CYTHON_TRACE=1)
    }
)

setup_options = {"build_ext": {"inplace": True}}

setup(
    options=setup_options,
    ext_modules=ext_modules,
    # install_requires=[
    #     'numpy',
    #     'scipy',
    #     'cython',
    #     'matplotlib',
    #     'pandas'
    # ],
    # python_requires='>=3.8',
)