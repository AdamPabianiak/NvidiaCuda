#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

void matrixMulCPU(int *a, int *b, int *c)
{
    int val = 0;

    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            val = 0;
            for (int k = 0; k < N; ++k)
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

int main()
{
    // Timing variables
    clock_t start, end;
    double cpu_time_used;

    // Start timer
    start = clock();

    int *a, *b, *c;

    int size = N * N * sizeof(int); // Number of bytes of an N x N matrix

    // Allocate memory using standard malloc
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize memory
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
        {
            a[row * N + col] = row;
            b[row * N + col] = col + 2;
            c[row * N + col] = 0;
        }

    // Perform matrix multiplication on CPU
    matrixMulCPU(a, b, c);

    // Verify the results (optional)
    printf("Calculation completed! Sample value c[10][10] = %d\n", c[10 * N + 10]);

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    // End timer
    end = clock();

    // Calculate time taken in seconds
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print timing information
    printf("Time taken: %f seconds\n", cpu_time_used);
    printf("Time taken: %f milliseconds\n", cpu_time_used * 1000);

    return 0;
}