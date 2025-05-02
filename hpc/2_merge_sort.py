import numpy as np
import time
import random
import multiprocessing


def merge(left, right):
    """Merge two sorted arrays"""
    n1, n2 = len(left), len(right)
    merged_arr = np.zeros(n1 + n2, dtype=int)
    i = j = 0

    for k in range(n1 + n2):
        if i == n1:
            merged_arr[k:] = right[j:]
            break
        elif j == n2:
            merged_arr[k:] = left[i:]
            break
        elif left[i] <= right[j]:
            merged_arr[k] = left[i]
            i += 1
        else:
            merged_arr[k] = right[j]
            j += 1

    return merged_arr


def sequential_merge_sort(arr):
    """Sequential merge sort implementation"""
    n = len(arr)

    # Base case
    if n <= 1:
        return arr

    # Split the array into two halves
    mid = n // 2
    left = arr[:mid]
    right = arr[mid:]

    # Sort each half recursively
    left_sorted = sequential_merge_sort(left)
    right_sorted = sequential_merge_sort(right)

    # Merge the two sorted halves
    return merge(left_sorted, right_sorted)


def parallel_sort_worker(chunk):
    """Worker function to sort a chunk of the array"""
    return sequential_merge_sort(chunk)


def parallel_merge_sort(arr, num_processes=None):
    """Parallel merge sort using process pool"""
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    n = len(arr)

    # For small arrays, use sequential sort
    if n <= 1000 or num_processes <= 1:
        return sequential_merge_sort(arr)

    # Split array into chunks for parallel processing
    chunk_size = n // num_processes
    chunks = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]

    # Sort chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        sorted_chunks = pool.map(parallel_sort_worker, chunks)

    # Merge sorted chunks
    result = sorted_chunks[0]
    for chunk in sorted_chunks[1:]:
        result = merge(result, chunk)

    return result


def main():
    # Generate a random array of 10,000 integers
    arr = np.array([random.randint(0, 100) for _ in range(10000)])
    print(f"Original array (first 10 elements): {arr[:10]}")

    start_time = time.time()
    sorted_arr = parallel_merge_sort(arr)
    end_time = time.time()

    print(f"Sorted array (first 10 elements): {sorted_arr[:10]}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Verify the sort is correct
    is_sorted = np.all(sorted_arr[:-1] <= sorted_arr[1:])
    print(f"Is correctly sorted: {is_sorted}")


if __name__ == "__main__":
    main()
