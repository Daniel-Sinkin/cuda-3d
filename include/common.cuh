#include <concepts>
#include <type_traits>

template <typename T>
concept CudaCompatible = std::is_arithmetic_v<T> &&
                         std::is_trivially_copyable_v<T> &&
                         (sizeof(T) <= 8);

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define CHECK_CUBLAS(call)                                                  \
    {                                                                       \
        cublasStatus_t status = call;                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuBLAS error in file '%s' in line %i : %d.\n", \
                __FILE__, __LINE__, status);                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#define PANIC(msg)                                                        \
    do {                                                                  \
        cudaDeviceSynchronize();                                          \
        fprintf(stderr, "PANIC at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        std::exit(EXIT_FAILURE);                                          \
    } while (0)