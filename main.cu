#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>


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
//确定图片位置（i，j）,然后计算该位置的最终颜色
__global__ void render(float* fb, int max_x, int max_y) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x * 3 + i * 3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main()
{
    //生成图片的边框大小 
    int num_pixels_x = 1200;//长
    int num_pixels_y = 800;//宽

    //在GPU上分配, threads_x * threads_y 个区块以进行计算,
    int threads_x = 8;
    int threads_y = 8;


    int num_pixels = num_pixels_x * num_pixels_y;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    //调用 frame buffer，在gpu上计算，然后传送到cpu上
    float* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Render our buffer
    //渲染FB
    dim3 blocks(num_pixels_x / threads_x + 1, num_pixels_y / threads_y + 1);
    dim3 threads(threads_x, threads_y);
    render << <blocks, threads >> > (fb, num_pixels_x, num_pixels_y);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 输出FB
    std::cout << "P3\n" << num_pixels_x << " " << num_pixels_y << "\n255\n";
    for (int j = num_pixels_y - 1; j >= 0; j--)
    {
        for (int i = 0; i < num_pixels_x; i++)
        {
            size_t pixel_index = j * 3 * num_pixels_x + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    checkCudaErrors(cudaFree(fb));
}


