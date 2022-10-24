#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"

//每次调用CUDA API都要检查返回的error code
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) 
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
/*
将光线方向缩放为单位长度，根据颜色混合以实现垂直渐变。并利用插值实现颜色的渐变
混合值=(1-t) * 起点值 + t * 终点值 ，t 为射线方向的单位向量缩放到y轴的值
*/
__device__ color3 ray_color(const ray& r)//, const hittable& world, int depth)
{
    //hit_record rec;

    ////限制递归次数
    //if (depth < 0)
    //{
    //    return color3(0, 0, 0);
    //}

    //if (world.hit(r, 0.001, infinity, rec))
    //{
    //    //漫反射
    //    //point3 target = rec.p + rec.normal + random_unit_hemisphere(rec.normal);//C 为交点外一随机圆的一点
    //    //return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);//折射光线即为 C-P(交点)

    //    ray scattered;//散射光
    //    color3 attenuation;//衰减值
    //    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
    //        return attenuation * ray_color(scattered, world, depth - 1);
    //    return color3(0, 0, 0);
    //}
    //auto t = hit_sphere(point3(0, 0, -1), 0.5, r);
    //if (t > 0.0) //确保是摄像机前方的物体
    //{
    //    vec3 Normal = unit_vec3(r.at(t) - vec3(0, 0, -1));
    //    return 0.5 * color3(Normal.x() + 1, Normal.y() + 1, Normal.z() + 1);
    //}
    //在（0，0，-1）添加一个圆，射线过则变红
    //if (hit_sphere(point3(0, 0, -1), 0.5, r))
    //    return  color3(1, 0, 0);

    //渐变背景
    vec3 unit_dirction = unit_vec3(r.direction());
    float t = 0.5f * (unit_dirction.y() + 1.0f);
    return (1.0f - t) * color3(1.0, 1.0, 1.0) + t * color3(0.5, 0.7, 1.0);
}


//确定图片位置（i，j）,然后计算该位置的最终颜色
__global__ void render(color3* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = ray_color(r);
}

int main()
{
    //生成图片的边框大小 
    int num_pixels_x = 1200;//长
    int num_pixels_y = 800;//宽
    //在GPU上分配, threads_x * threads_y 个区块以进行计算,
    int threads_x = 8;
    int threads_y = 8;

    std::cerr << "Rendering a " << num_pixels_x << "x" << num_pixels_y << " image ";
    std::cerr << "in " << threads_x << "x" << threads_y << " blocks.\n";

    int num_pixels = num_pixels_x * num_pixels_y;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    //调用 frame buffer，在gpu上计算，然后传送到cpu上
    color3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, stop;//记时间
    start = clock();
    // Render our buffer
    //渲染FB
    dim3 blocks(num_pixels_x / threads_x + 1, num_pixels_y / threads_y + 1);
    dim3 threads(threads_x, threads_y);
    render << <blocks, threads >> > (fb, num_pixels_x, num_pixels_y,
                                        vec3(-2.0, -1.0, -1.0),
                                        vec3(4.0, 0.0, 0.0),
                                        vec3(0.0, 2.0, 0.0),
                                        vec3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // 输出FB
    std::cout << "P3\n" << num_pixels_x << " " << num_pixels_y << "\n255\n";
    for (int j = num_pixels_y - 1; j >= 0; j--)
    {
        for (int i = 0; i < num_pixels_x; i++)
        {
            size_t pixel_index = j * num_pixels_x + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
}


