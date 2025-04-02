#include <stdio.h>
#include <time.h>

void initWith(float num, float *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = num;
    }
}

void addVectorsInto(float *result, float *a, float *b, int N)
{
    for (int i = 0; i < N; ++i)
    {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    // Timing variables
    clock_t start, end;
    double cpu_time_used;

    // Start timer
    start = clock();

    const int N = 2 << 28;
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    addVectorsInto(c, a, b, N);

    checkElementsAre(7, c, N);

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
}
