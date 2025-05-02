import numpy as np
from numba import cuda
import math
import time

# Define the CUDA kernel for matrix multiplication
@cuda.jit
def matrix_mul_kernel(A, B, C):
    """
    CUDA kernel for matrix multiplication.
    Each thread computes one element of the output matrix C.
    C = A * B
    """
    # Get thread indices
    row, col = cuda.grid(2)
    
    # Check if thread is within bounds
    if row < C.shape[0] and col < C.shape[1]:
        # Initialize the result element to zero
        tmp = 0.0
        
        # Loop over elements in the row of A and column of B
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
            
        # Write the result to global memory
        C[row, col] = tmp

def matrix_multiply_cuda(A, B):
    """
    Performs matrix multiplication using CUDA.
    Args:
        A: First matrix (numpy array)
        B: Second matrix (numpy array)
    Returns:
        C: Result matrix (numpy array)
    """
    # Check if matrices can be multiplied
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Cannot multiply matrices of shapes {A.shape} and {B.shape}")
    
    # Create output array
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    
    # Copy arrays to device
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    
    # Set up the grid and block dimensions
    threads_per_block = (16, 16)  # 256 threads per block
    blocks_per_grid_x = math.ceil(C.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(C.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch the kernel
    matrix_mul_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    
    # Copy the result back to the host
    cuda.synchronize()  # Make sure kernel execution is finished
    C = d_C.copy_to_host()
    
    return C

# Example usage
def main():
    # Create sample matrices
    matrix_size = 1000
    A = np.random.random((matrix_size, matrix_size)).astype(np.float32)
    B = np.random.random((matrix_size, matrix_size)).astype(np.float32)
    
    # Time the CUDA implementation
    start_time = time.time()
    C_cuda = matrix_multiply_cuda(A, B)
    cuda_time = time.time() - start_time
    print(f"CUDA matrix multiplication time: {cuda_time:.4f} seconds")
    
    # Time the NumPy implementation for comparison
    start_time = time.time()
    C_numpy = np.matmul(A, B)
    numpy_time = time.time() - start_time
    print(f"NumPy matrix multiplication time: {numpy_time:.4f} seconds")
    
    # Verify the results match
    print(f"Results match: {np.allclose(C_cuda, C_numpy)}")
    print(f"Speedup: {numpy_time / cuda_time:.2f}x")

if __name__ == "__main__":
    main()