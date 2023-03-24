#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "lib_matmul.h"

int main(int argc, char *argv[]) {
    size_t dim = atoi(argv[1]);

  float *A = malloc(dim * dim * sizeof(float));
  float *B = malloc(dim * dim * sizeof(float));
  float *C = malloc(dim * dim * sizeof(float));

  for (size_t i = 0; i < dim * dim; i++) {
    A[i] = rand() % 5;
    B[i] = rand() % 5;
  }

  // your code or function here
    naive_mul(A, B, C, dim);
// you can activate check by adding -DCHECK_MUL to your command line
#ifdef CHECK_MUL
printf("Checking result...\n");
  check(A, B, C, dim);
#endif

  free(A);
  free(B);
  free(C);

  return 0;
}
