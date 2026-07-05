import ctypes
import os
import sys

class MatrixBridge:
    def __init__(self):
        if not os.path.exists("./libmatrix_acc.so") and not os.path.exists("./libmatrix_acc.dll"):
            if sys.platform.startswith("win"):
                os.system("gcc -shared -o libmatrix_acc.dll simd_accelerator.c")
                lib_path = "./libmatrix_acc.dll"
            else:
                os.system("gcc -shared -fPIC -o libmatrix_acc.so simd_accelerator.c")
                lib_path = "./libmatrix_acc.so"
        else:
            lib_path = "./libmatrix_acc.dll" if sys.platform.startswith("win") else "./libmatrix_acc.so"

        self.lib = ctypes.CDLL(lib_path)
        self.lib.compute_cosine_similarity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        self.lib.compute_cosine_similarity.restype = ctypes.c_float

    def calculate_similarity(self, embed_a: list, embed_b: list) -> float:
        c_array_a = (ctypes.c_float * len(embed_a))(*embed_a)
        c_array_b = (ctypes.c_float * len(embed_b))(*embed_b)
        return self.lib.compute_cosine_similarity(c_array_a, c_array_b)
