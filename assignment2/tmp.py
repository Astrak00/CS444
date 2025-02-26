import numpy as np
import time
import os
import sys
SIZE = int(sys.argv[1])

# Create two 9x9 matrices with random values
matrix1 = np.random.random(size=(SIZE, SIZE))
matrix2 = np.random.random(size=(SIZE, SIZE))

start = time.time()
# Perform matrix multiplication using Numpy's matmul function
result_matrix = np.matmul(matrix1, matrix2)
end = time.time()
print(f"Time taken for a {SIZE}x{SIZE} matrix is: {end-start}")
