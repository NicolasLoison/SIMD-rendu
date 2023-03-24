#include <stdio.h>
#include <immintrin.h>
#include "../src/lib_matmul.h"

int main() {
    size_t dim = 512;

    float *A = _mm_malloc(dim * dim * sizeof(float), 16);
    float *B = _mm_malloc(dim * dim * sizeof(float), 16);
    float *C = _mm_malloc(dim * dim * sizeof(float), 16);

    for (size_t i = 0; i < dim * dim; i++) {
        A[i] = rand() % 5;
        B[i] = rand() % 5;
    }

    // your code or function here
    naive_mul(A, B, C, dim);

    check(A,B,C,dim);


    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
