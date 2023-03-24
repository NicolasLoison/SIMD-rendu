#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lib_matmul.h"

int main(int argc, char *argv[]) {
    size_t dim = atoi(argv[1]);

    float *A = _mm_malloc(dim * dim * sizeof(float), 16);
    float *B = _mm_malloc(dim * dim * sizeof(float), 16);
    float *C = _mm_malloc(dim * dim * sizeof(float), 16);

  for (size_t i = 0; i < dim * dim; i++) {
    A[i] = rand() % 5;
    B[i] = rand() % 5;
  }

  // your code or function here
    sse_mul8(A, B, C, dim);
// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
  check(A, B, C, dim);
#endif

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

  return 0;
}
