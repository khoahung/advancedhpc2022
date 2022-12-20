import sys
import numba
from numba import cuda
import numpy as np



print("Python version:", sys.version)
print("Numba version:", numba.__version__)
print("Numpy version:", np.__version__)
print("Numpy version:",cuda.gpus.__module__)