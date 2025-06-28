#include "common.cuh"

template <CudaCompatible T>
class CuArray {
public:
    CuArray(size_t size) : size_(size), ptr_(nullptr) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_ * sizeof(T)));
    }

    ~CuArray() {
        if (ptr_) cudaFree(ptr_);
    }

    T *get() const { return ptr_; }
    T *operator->() const { return ptr_; }
    T &operator*() const { return *ptr_; }

    __host__ __device__
        size_t
        get_size() const { return size_; }

    __device__ T &operator[](size_t idx) const { return ptr_[idx]; }

    void copy_from_host(const T *h_data, size_t count) {
        if (count > size_) PANIC("count too big");
        CHECK_CUDA(cudaMemcpy(ptr_, h_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_from_host(const vector<T> &h_data) {
        copy_from_host(h_data.data(), h_data.size());
    }
    void copy_to_host(T *h_data, size_t count) const {
        CHECK_CUDA(cudaMemcpy(h_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copy_to_host(T *h_data) const {
        CHECK_CUDA(cudaMemcpy(h_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void copy_to_host(vector<T> &h_data) const {
        if (h_data.size() != size_) PANIC("host vector size mismatch");
        copy_to_host(h_data.data(), size_);
    }

    // Implicit copy is not allowed
    CuArray(const CuArray &) = delete;
    CuArray &operator=(const CuArray &) = delete;

    // Move is allowed
    CuArray(CuArray &&other) noexcept
        : size_(other.size_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    // Move assignment
    CuArray &operator=(CuArray &&other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

private:
    size_t size_;
    T *ptr_;
};