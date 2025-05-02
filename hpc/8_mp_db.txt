import multiprocessing
import itertools
import time
import random

# Define your table statistics dynamically (you can modify this)
TABLES = {
    "Customers": 100_000,
    "Orders": 500_000,
    "Products": 10_000,
    "Suppliers": 1_000
}

# Simulated cost function for join order (mocked with randomness + delay)
def estimate_cost(join_order):
    time.sleep(0.1)  # simulate computation delay
    cost = 0
    for i in range(len(join_order) - 1):
        t1 = TABLES[join_order[i]]
        t2 = TABLES[join_order[i + 1]]
        selectivity = random.randint(100, 1000)
        cost += (t1 * t2) // selectivity
    return (join_order, cost)

def run_sequential(permutations):
    start = time.time()
    results = [estimate_cost(p) for p in permutations]
    end = time.time()
    return results, end - start

def run_parallel(permutations):
    start = time.time()
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(estimate_cost, permutations)
    end = time.time()
    return results, end - start

def query_optimizer():
    tables = list(TABLES.keys())
    permutations = list(itertools.permutations(tables))

    print(f"\n🔢 Total Join Orders to Evaluate: {len(permutations)}")

    # Sequential Evaluation
    print("\n⏳ Running Sequential Optimization...")
    seq_results, seq_time = run_sequential(permutations)
    best_seq = min(seq_results, key=lambda x: x[1])

    # Parallel Evaluation
    print("\n🚀 Running Parallel Optimization...")
    par_results, par_time = run_parallel(permutations)
    best_par = min(par_results, key=lambda x: x[1])

    # Output Results
    print("\n📊 Optimization Summary:")
    print(f"Sequential Time: {seq_time:.2f} seconds")
    print(f"Parallel Time:   {par_time:.2f} seconds")
    print("\n✅ Best Join Order Found (Parallel):")
    print(f"Join Order: {' -> '.join(best_par[0])}")
    print(f"Estimated Cost: {best_par[1]}")

if __name__ == "__main__":
    query_optimizer()
