#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define CHUNK_SIZE 1000

struct ChunkStats
{
    int min_val;
    int sum_val;
    int size;
};

struct ChunkStats get_chunk_stats(int *chunk, int chunk_size)
{
    struct ChunkStats stats;
    stats.min_val = chunk[0];
    stats.sum_val = 0;
    stats.size = chunk_size;

    for (int i = 0; i < chunk_size; i++)
    {
        if (chunk[i] < stats.min_val)
        {
            stats.min_val = chunk[i];
        }
        stats.sum_val += chunk[i];
    }

    return stats;
}

void parallel_reduction_min_avg(int *data, int data_size, int *min_val_ptr, double *avg_val_ptr)
{
    int num_threads = omp_get_max_threads();
    int chunk_size = data_size / num_threads;
    int num_chunks = num_threads;

    if (data_size % chunk_size != 0)
    {
        num_chunks++;
    }

    struct ChunkStats *chunk_stats = malloc(num_chunks * sizeof(struct ChunkStats));

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int start_index = thread_id * chunk_size;
        int end_index = (thread_id + 1) * chunk_size - 1;

        if (thread_id == num_threads - 1)
        {
            end_index = data_size - 1;
        }

        int chunk_size_actual = end_index - start_index + 1;
        int *chunk = data + start_index;

        chunk_stats[thread_id] = get_chunk_stats(chunk, chunk_size_actual);

#pragma omp barrier

        for (int i = 1, j = thread_id - 1; i <= num_threads && j >= 0; i *= 2, j -= i)
        {
            if (thread_id % i == 0 && thread_id + i < num_threads)
            {
                chunk_stats[thread_id].min_val = (chunk_stats[thread_id].min_val < chunk_stats[thread_id + i].min_val) ? chunk_stats[thread_id].min_val : chunk_stats[thread_id + i].min_val;
                chunk_stats[thread_id].sum_val += chunk_stats[thread_id + i].sum_val;
                chunk_stats[thread_id].size += chunk_stats[thread_id + i].size;
            }
#pragma omp barrier
        }
    }

    int min_val = chunk_stats[0].min_val;
    int sum_val = chunk_stats[0].sum_val;
    int size = chunk_stats[0].size;

    for (int i = 1, j = 0; i < num_chunks; i *= 2, j++)
    {
        if (j % i == 0 && j + i < num_chunks)
        {
            min_val = (min_val < chunk_stats[j + i].min_val) ? min_val : chunk_stats[j + i].min_val;
            sum_val += chunk_stats[j + i].sum_val;
            size += chunk_stats[j + i].size;
        }
    }

    *min_val_ptr = min_val;
    *avg_val_ptr = (double)sum_val / (double)size;

    free(chunk_stats);
}

int main()
{
    int data_size = 1000000;
    int *data = malloc(data_size * sizeof(int));

    for (int i = 0; i < data_size; i++)
    {
        data[i] = rand() % 100;
    }

    int min_val;
    double avg_val;

    parallel_reduction_min_avg(data, data_size, &min_val, &avg_val);

    printf("Minimum value: %d\n", min_val);
    printf("Average value: %lf\n", avg_val);

    free(data);
    return 0;
}
