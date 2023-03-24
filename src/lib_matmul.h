#include <immintrin.h>
#include <xmmintrin.h>

void naive_mul(float *A, float *B, float *res, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            int sum = 0;
            for (size_t k = 0; k < dim; k++) {
                sum += A[i * dim + k] * B[k * dim + j];
            }
            res[i * dim + j] = sum;
        }
    }
}

void naive_mul2(float *A, float *B, float *res, size_t dim) {
    for (size_t i = 0; i < dim; i++){
        for (size_t j = 0; j < dim; j++){
            float Acell = A[i * dim + j];
            for (size_t k = 0; k < dim; k++){
                res[i*dim+k] += Acell * B[j * dim + k];
            }
        }
    }
}

void check(float *A, float *B, float *res, size_t dim) {
    float *exp = malloc(dim * dim * sizeof(float));
    naive_mul(A, B, exp, dim);

    for (size_t i = 0; i < dim * dim; i++) {
        if (exp[i] != res[i]) {
            printf("Value at %lu differs: %f\n", i, exp[i] - res[i]);
            exit(1);
        }
    }
}

void sse_mul(float *A, float *B, float *res, size_t dim){
    for (size_t i = 0; i < dim; i++){
        for (size_t j = 0; j < dim; j++){
            float Acell = A[i * dim + j];
            for (size_t k = 0; k < dim; k+=4){
                __m128 vec = _mm_load_ps(&res[i*dim+k]);
                __m128 vec2 = _mm_load_ps(&B[j * dim + k]);
                vec = _mm_add_ps(vec, _mm_mul_ps(_mm_set1_ps(Acell), vec2));
                _mm_store_ps(&res[i*dim+k], vec);
            }
        }
    }
}

void sse_mul8(float *A, float *B, float *res, size_t dim){
    for (size_t i = 0; i < dim; i++){
        for (size_t j = 0; j < dim; j++){
            float Acell = A[i * dim + j];
            for (size_t k = 0; k < dim; k+=8){
                __m128 vec = _mm_load_ps(&res[i*dim+k]);
                __m128 vecc = _mm_load_ps(&res[i*dim+k+4]);
                __m128 vec2 = _mm_load_ps(&B[j * dim + k]);
                __m128 vecc2 = _mm_load_ps(&B[j*dim+k+4]);
                vec = _mm_add_ps(vec, _mm_mul_ps(_mm_set1_ps(Acell), vec2));
                vecc = _mm_add_ps(vecc, _mm_mul_ps(_mm_set1_ps(Acell), vecc2));
                _mm_store_ps(&res[i*dim+k], vec);
                _mm_store_ps(&res[i*dim+k+4], vecc);
            }
        }
    }
}

void sse_mul16(float *A, float *B, float *res, size_t dim){
    for (size_t i = 0; i < dim; i++){
        for (size_t j = 0; j < dim; j++){
            float Acell = A[i * dim + j];
            for (size_t k = 0; k < dim; k+=16){
                __m128 vec = _mm_load_ps(&res[i*dim+k]);
                __m128 vecc = _mm_load_ps(&res[i*dim+k+4]);
                __m128 veccc = _mm_load_ps(&res[i*dim+k+8]);
                __m128 vecccc = _mm_load_ps(&res[i*dim+k+12]);
                __m128 vec2 = _mm_load_ps(&B[j * dim + k]);
                __m128 vecc2 = _mm_load_ps(&B[j*dim+k+4]);
                __m128 veccc2 = _mm_load_ps(&B[j * dim + k+8]);
                __m128 vecccc2 = _mm_load_ps(&B[j*dim+k+12]);
                vec = _mm_add_ps(vec, _mm_mul_ps(_mm_set1_ps(Acell), vec2));
                vecc = _mm_add_ps(vecc, _mm_mul_ps(_mm_set1_ps(Acell), vecc2));
                veccc = _mm_add_ps(veccc, _mm_mul_ps(_mm_set1_ps(Acell), veccc2));
                vecccc = _mm_add_ps(vecccc, _mm_mul_ps(_mm_set1_ps(Acell), vecccc2));
                _mm_store_ps(&res[i*dim+k], vec);
                _mm_store_ps(&res[i*dim+k+4], vecc);
                _mm_store_ps(&res[i*dim+k+8], veccc);
                _mm_store_ps(&res[i*dim+k+12], vecccc);
            }
        }
    }
}