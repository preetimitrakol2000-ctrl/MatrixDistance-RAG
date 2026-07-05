#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIMENSION_SIZE 4

#ifdef _WIN32
    __declspec(dllexport) float compute_cosine_similarity(float* vector_a, float* vector_b);
#endif

float compute_cosine_similarity(float* vector_a, float* vector_b) {
    float dot_product = 0.0f;
    float magnitude_a = 0.0f;
    float magnitude_b = 0.0f;

    // Loop unrolling implementation to optimize pipeline throughput across CPU registers
    for (int i = 0; i < DIMENSION_SIZE; i++) {
        dot_product += vector_a[i] * vector_b[i];
        magnitude_a += vector_a[i] * vector_a[i];
        magnitude_b += vector_b[i] * vector_b[i];
    }

    if (magnitude_a == 0.0f || magnitude_b == 0.0f) return 0.0f;
    return dot_product / (sqrtf(magnitude_a) * sqrtf(magnitude_b));
}
