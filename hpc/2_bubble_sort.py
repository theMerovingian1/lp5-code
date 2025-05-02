import numpy as np
import time
import random
import multiprocessing


def parallel_bubble_sort(arr):
    n = len(arr)
    # Odd-even sort algorithm (parallel-friendly version of bubble sort)
    for i in range(n):
        # Alternate between odd and even phases
        start = i % 2

        # Use multiprocessing Pool as Python doesn't have direct OpenMP support
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Process pairs in parallel
            # For each phase, we work on non-overlapping pairs
            pairs = [(j, j+1)
                     for j in range(start, n-1, 2) if arr[j] > arr[j+1]]

            # Swap elements if needed
            for j, j_plus_1 in pairs:
                arr[j], arr[j_plus_1] = arr[j_plus_1], arr[j]

    return arr


def main():
    # Generate a random array of 100 integers
    arr_size = 100
    print(f"Generating array of SIZE = {arr_size} integers:")
    arr = np.array([random.randint(0, 100) for _ in range(arr_size)])
    print(f"Original array (first 10 elements): {arr[:10]}")

    start_time = time.time()
    parallel_bubble_sort(arr)
    end_time = time.time()

    print(f"Sorted array (first 10 elements): {arr[:10]}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
