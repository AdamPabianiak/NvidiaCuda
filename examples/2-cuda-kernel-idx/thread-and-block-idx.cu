#include <stdio.h>

__global__ void printHelloForCorrectExecutionConfiguration()
{

    if (threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Hello from GPU thread!\n");
    }
}

int main()
{
    // TODO: Change kernel parameters to print Hello message
    printHelloForCorrectExecutionConfiguration<<<2, 4>>>();

    cudaDeviceSynchronize();
}
