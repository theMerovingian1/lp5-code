from mpi4py import MPI
import numpy as np
import random
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def quicksort_serial(arr):
    """Serial implementation of quicksort algorithm"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort_serial(left) + middle + quicksort_serial(right)


def quicksort_parallel(arr):
    """Parallel implementation of quicksort algorithm using MPI"""
    # Root process (rank 0) does the initial partitioning
    if rank == 0:
        if len(arr) <= 1:
            return arr

        pivot = arr[len(arr) // 2]
        left = np.array([x for x in arr if x < pivot], dtype=np.int32)
        middle = np.array([x for x in arr if x == pivot], dtype=np.int32)
        right = np.array([x for x in arr if x > pivot], dtype=np.int32)

        # Calculate sizes to distribute work evenly
        left_chunks = np.array_split(left, size) if len(left) > 0 else [
            np.array([], dtype=np.int32)] * size
        right_chunks = np.array_split(right, size) if len(right) > 0 else [
            np.array([], dtype=np.int32)] * size
    else:
        left_chunks = None
        right_chunks = None
        middle = None

    # Broadcast the middle elements to all processes
    middle = comm.bcast(middle, root=0)

    # Scatter the left and right arrays to all processes
    local_left = comm.scatter(left_chunks, root=0)
    local_right = comm.scatter(right_chunks, root=0)

    # Sort locally
    sorted_left = quicksort_serial(local_left)
    sorted_right = quicksort_serial(local_right)

    # Gather results back to root
    all_left = comm.gather(sorted_left, root=0)
    all_right = comm.gather(sorted_right, root=0)

    # Combine results at root process
    if rank == 0:
        final_left = np.concatenate(
            [chunk for chunk in all_left if len(chunk) > 0])
        final_right = np.concatenate(
            [chunk for chunk in all_right if len(chunk) > 0])
        return np.concatenate((final_left, middle, final_right))

    return None


def main():
    # Only root process generates data
    if rank == 0:
        # Generate a dataset (reduced size for testing)
        np.random.seed(42)  # For reproducibility
        arr_size = 1000000
        arr = np.random.randint(0, 1000, size=arr_size, dtype=np.int32)
        print(f"Generated array of size {arr_size}")

        # Make a copy for serial sorting
        arr_copy = arr.copy()

        # Time the serial version
        start_time = time.time()
        sorted_serial = quicksort_serial(arr_copy)
        serial_time = time.time() - start_time
        print(f"Serial Quicksort completed in {serial_time:.4f} seconds")
    else:
        arr = None

    # Time the parallel version
    comm.barrier()  # Synchronize all processes before timing

    start_time = time.time()
    sorted_parallel = quicksort_parallel(arr)
    comm.barrier()  # Wait for all processes to complete

    parallel_time = time.time() - start_time

    # Print results from root process
    if rank == 0:
        print(f"Parallel Quicksort completed in {parallel_time:.4f} seconds")
        print(f"Speedup: {serial_time / parallel_time:.2f}x")


if __name__ == "__main__":
    main()
