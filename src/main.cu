#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE);                                                                   \
        }                                                                                              \
    } while (0)

namespace Config {
constexpr int width = 1280;
constexpr int height = 720;
constexpr int num_pixels = width * height;

constexpr float aspect_ratio = float(width) / float(height);
} // namespace Config

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x = 0.0, float y = 0.0, float z = 0.0) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3(const Vec3 &other) : x(other.x), y(other.y), z(other.z) {}
    __host__ __device__ Vec3(Vec3 &&other) noexcept : x(other.x), y(other.y), z(other.z) {
        other.x = other.y = other.z = 0.0f;
    }
    __host__ __device__ Vec3 &operator=(const Vec3 &other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }
    __host__ __device__ Vec3 &operator=(Vec3 &&other) noexcept {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
            other.x = other.y = other.z = 0.0f;
        }
        return *this;
    }
    [[nodiscard]] __host__ __device__ Vec3 operator+(const Vec3 &v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    [[nodiscard]] __host__ __device__ Vec3 operator-(const Vec3 &v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    [[nodiscard]] __host__ __device__ Vec3 operator*(float t) const {
        return Vec3(x * t, y * t, z * t);
    }
    [[nodiscard]] __host__ __device__ Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }
    __host__ __device__ Vec3 &operator+=(const Vec3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ Vec3 &operator-=(const Vec3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ Vec3 &operator*=(float t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }
    [[nodiscard]] __host__ __device__ float dot(const Vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    [[nodiscard]] __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
    [[nodiscard]] __host__ __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }
    [[nodiscard]] __host__ __device__ Vec3 normalize() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
    __host__ __device__ void normalize_inplace() {
        float len = length();
        if (len > 0) {
            x /= len;
            y /= len;
            z /= len;
        } else {
            x = y = z = 0.0f;
        }
    }
};

__device__ Vec3 lerp(const Vec3 &a, const Vec3 &b, float k) {
    // k must be in [0, 1] for this to make sense
    return a * (1.0f - k) + b * k;
}

struct Sphere {
    Vec3 origin;
    float radius;
    Vec3 color;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__ Ray(const Vec3 &o, const Vec3 &d) : origin(o), direction(d) {}
};

enum class ColliderType {
    Sphere,
    // Box,
    // Plane
};

struct Collider {
    ColliderType type;
    Sphere sphere;

    __host__ __device__ Collider() : type(ColliderType::Sphere), sphere() {}
    __host__ __device__ Collider(const Sphere &s) : type(ColliderType::Sphere), sphere(s) {}
};

struct Object {
    Collider collider;
    int id;
    Vec3 color;

    __host__ __device__ Object() : collider(), id(-1), color(1, 1, 1) {}
    __host__ __device__ Object(const Collider &col, int object_id, Vec3 object_color = Vec3(1, 1, 1))
        : collider(col), id(object_id), color(object_color) {}
};

__device__ Object *d_objects;
__device__ int d_num_objects;

constexpr int pixel_buffer_size = Config::num_pixels * sizeof(Pixel);

__host__ __device__ float to_ndc(float x) { return 2.0f * x - 1.0f; }
__host__ __device__ double to_ndc(double x) { return 2.0 * x - 1.0; }

__host__ __device__ float to_ndc_with_aspect(float x) { return to_ndc(x) * Config::aspect_ratio; }
__host__ __device__ double to_ndc_with_aspect(double x) { return to_ndc(x) * static_cast<double>(Config::aspect_ratio); }

__device__ Vec3 normal_to_color(const Vec3 &normal) {
    // Map normal components from [-1, 1] to [0, 1] for RGB
    float r = (normal.x + 1.0f) * 0.5f;
    float g = (normal.y + 1.0f) * 0.5f;
    float b = (normal.z + 1.0f) * 0.5f;

    return Vec3(r, g, b);
}

struct HitInfo {
    bool has_hit;
    float hit_angle;
    Vec3 hit_point;
    Vec3 normal;
    float distance;
    Vec3 color;
    size_t obj_id;

    __host__ __device__ HitInfo(float angle, Vec3 point, Vec3 norm, float dist, Vec3 color_, size_t obj_id_)
        : has_hit(true), hit_angle(angle), hit_point(point), normal(norm), distance(dist), color(color_), obj_id(obj_id_) {}
    __host__ __device__ HitInfo(float angle, Vec3 point, Vec3 norm, float dist, Vec3 color_)
        : has_hit(true), hit_angle(angle), hit_point(point), normal(norm), distance(dist), color(color_), obj_id(0) {}
    __host__ __device__ HitInfo(float angle, Vec3 point, Vec3 norm, float dist)
        : has_hit(true), hit_angle(angle), hit_point(point), normal(norm), distance(dist), color(Vec3()), obj_id(0) {}
    __host__ __device__ static HitInfo no_hit() {
        HitInfo info;
        info.has_hit = false;
        info.hit_angle = 0.0f;
        info.hit_point = Vec3(0, 0, 0);
        info.normal = Vec3(0, 0, 0);
        info.distance = 0.0f;
        return info;
    }

private:
    __host__ __device__ HitInfo() = default;
};

__device__ HitInfo ray_sphere_collision(Ray ray, Sphere sphere) {
    Vec3 oc = ray.origin - sphere.origin;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;

    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0) {
        return HitInfo::no_hit();
    }

    float sqrt_discriminant = sqrtf(discriminant);
    float dist = (-b - sqrt_discriminant) / (2.0f * a);

    if (dist < 0.0f) {
        dist = (-b + sqrt_discriminant) / (2.0f * a);
        if (dist < 0.0f) {
            return HitInfo::no_hit();
        }
    }

    Vec3 hit_point = ray.origin + ray.direction * dist;
    Vec3 normal = (hit_point - sphere.origin).normalize();

    float cos_angle = fabsf(ray.direction.dot(normal));
    float hit_angle = acosf(cos_angle);

    return HitInfo(hit_angle, hit_point, normal, dist);
}

__device__ HitInfo hit_sphere(float x, float y) {
    float u = to_ndc_with_aspect(x);
    float v = to_ndc(y);

    Vec3 sphere_center(0.0f, 0.0f, 0.0f);
    float sphere_radius = 0.5f;

    Ray ray(Vec3(u, v, 3.0f), Vec3(0.0f, 0.0f, -1.0f));
    Sphere sphere{sphere_center, sphere_radius, Vec3(1.0f, 1.0f, 1.0f)};

    return ray_sphere_collision(ray, sphere);
}

__device__ HitInfo hit_scene(float x, float y) {
    float u = to_ndc_with_aspect(x);
    float v = to_ndc(y);

    Ray ray(Vec3(u, v, 3.0f), Vec3(0.0f, 0.0f, -1.0f));

    HitInfo closest_hit = HitInfo::no_hit();
    float min_distance = FLT_MAX;

    for (int i = 0; i < d_num_objects; i++) {
        const Object &obj = d_objects[i];
        if (obj.collider.type == ColliderType::Sphere) {
            const Sphere &sphere = obj.collider.sphere;
            HitInfo hit_info = ray_sphere_collision(ray, sphere);
            if (hit_info.has_hit && hit_info.distance < min_distance) {
                min_distance = hit_info.distance;
                hit_info.obj_id = obj.id;
                hit_info.color = obj.color;
                closest_hit = hit_info;
            }
        }
    }
    return closest_hit;
}

__device__ float3 d_light_dir;

__global__ void per_pixel(Pixel *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    int pixel_idx = row * width + col;

    float u = float(col) / float(width);
    float v = float(row) / float(height);

    HitInfo hit_info = hit_scene(u, v);

    if (hit_info.has_hit) {
        float3 Lf = d_light_dir;
        Vec3 L(Lf.x, Lf.y, Lf.z);

        Vec3 N = hit_info.normal;

        const float ambient = 0.20f;
        float lambert = fmaxf(0.0f, N.dot(-L));
        float intensity = ambient + (1.0f - ambient) * lambert;
        Vec3 shaded = hit_info.color * intensity;
        shaded.x = fminf(shaded.x, 1.0f);
        shaded.y = fminf(shaded.y, 1.0f);
        shaded.z = fminf(shaded.z, 1.0f);
        output[pixel_idx].r = static_cast<unsigned char>(shaded.x * 255.0f);
        output[pixel_idx].g = static_cast<unsigned char>(shaded.y * 255.0f);
        output[pixel_idx].b = static_cast<unsigned char>(shaded.z * 255.0f);
    } else {
        output[pixel_idx].r = static_cast<unsigned char>(25);
        output[pixel_idx].g = static_cast<unsigned char>(12);
        output[pixel_idx].b = static_cast<unsigned char>(37);
    }
};

void savePPM(const char *filename, Pixel *data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);

    file << "P6\n"
         << width << " " << height << "\n255\n";

    file.write(reinterpret_cast<char *>(data), width * height * sizeof(Pixel));
}

int main(int argc, char **argv) {
    float3 h_light_dir = make_float3(-0.6f, -1.0f, -0.3f);
    float len = sqrtf(h_light_dir.x * h_light_dir.x +
                      h_light_dir.y * h_light_dir.y +
                      h_light_dir.z * h_light_dir.z);
    h_light_dir.x /= len;
    h_light_dir.y /= len;
    h_light_dir.z /= len;

    CUDA_CHECK(cudaMemcpyToSymbol(d_light_dir, &h_light_dir, sizeof(float3)));

    std::vector<Object> h_objects;

    // Initialize sphere color in the Sphere struct
    Sphere sphere1{Vec3(0.0f, 0.0f, 0.0f), 0.5f, Vec3(1.0f, 1.0f, 1.0f)};
    h_objects.push_back(Object(Collider(sphere1), 100, Vec3(1.0f, 1.0f, 1.0f)));

    Sphere sphere2{Vec3(-1.2f, 0.0f, 0.0f), 0.4f, Vec3(1.0f, 0.5f, 0.5f)};
    h_objects.push_back(Object(Collider(sphere2), 200, Vec3(1.0f, 0.5f, 0.5f)));

    Sphere sphere3{Vec3(1.2f, 0.0f, 0.0f), 0.4f, Vec3(0.5f, 1.0f, 0.5f)};
    h_objects.push_back(Object(Collider(sphere3), 300, Vec3(0.5f, 1.0f, 0.5f)));

    Sphere sphere4{Vec3(0.0f, -0.8f, -0.5f), 0.3f, Vec3(0.5f, 0.5f, 1.0f)};
    h_objects.push_back(Object(Collider(sphere4), 400, Vec3(0.5f, 0.5f, 1.0f)));

    Sphere sphere5{Vec3(0.0f, 0.8f, -0.5f), 0.3f, Vec3(1.0f, 1.0f, 0.5f)};
    h_objects.push_back(Object(Collider(sphere5), 500, Vec3(1.0f, 1.0f, 0.5f)));

    Object *objects_on_device;
    CUDA_CHECK(cudaMalloc(&objects_on_device, h_objects.size() * sizeof(Object)));
    CUDA_CHECK(cudaMemcpy(objects_on_device, h_objects.data(),
        h_objects.size() * sizeof(Object), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_objects, &objects_on_device, sizeof(Object *)));
    int num_objects = h_objects.size();
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_objects, &num_objects, sizeof(int)));

    Pixel *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, pixel_buffer_size));

    dim3 block_size(16, 16);
    int grid_size_x = (Config::width + block_size.x - 1) / block_size.x;
    int grid_size_y = (Config::height + block_size.y - 1) / block_size.y;
    dim3 grid_size(grid_size_x, grid_size_y);

    per_pixel<<<grid_size, block_size>>>(d_output, Config::width, Config::height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<Pixel> h_output(Config::num_pixels);

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, pixel_buffer_size, cudaMemcpyDeviceToHost));

    auto filename = "output.ppm";
    savePPM(filename, h_output.data(), Config::width, Config::height);

    std::cout << "Image saved as " << filename << "\n";

    std::cout << "Created " << h_objects.size() << " objects with IDs: ";
    for (const auto &obj : h_objects) {
        std::cout << obj.id << " ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(objects_on_device));

    return 0;
}