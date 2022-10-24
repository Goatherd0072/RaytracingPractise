#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>

#include"color.cuh"
#include"sphere.cuh"
#include"hittable_list.cuh"
#include"camera.cuh"

#include"vec3.cuh"
#include"material.cuh"
/*
这里的host端就是指CPU，device端就是指GPU；
使用__global__声明的核函数是在CPU端调用，在GPU里执行；__device__声明的函数调用和执行都在GPU中；__host__声明的函数调用和执行都在CPU端。
*/


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
圆的方程： x^2 + y^2 + z^2 = r^2      |   (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2 判断一个点是否在圆上则为两点的距离（目标点和圆心）与半径r的比较
(P-C) * (P-C) = (x-x0)^2 + (y-y0)^2 + (z-z0)^2  P为任意一点 C为圆心(x0,y0,z0)，(P-C)则为点到圆心的向量，代入 射线方程 则为 （P0+tU-C）
判断点是否在圆上即得 （P0+tU-C）^2 = r^2  ① ==> ((P0-C)+tU)^2 = r^2  ②
==> (P0-C)^2 + 2(P0-C)•tU + (tU)^2 - r^2 = 0  >> U^2•t^2 + 2(P0-C)U•t +((P0-C)^2 - r^2)  ③
③方程中 P0为需要判断的点，C为圆心坐标，U为射线方向，r为圆半径，t为未知量。t的不同会使得方程的解不同。
给定任意t与③方程的解的个数即为该射线在圆上的交点个数。
*/
__device__ bool hit_sphere(const point3& center, float radius, const ray& r)
{
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
    //vec3 P_C = p.origin() - center;//(P0-C)
    //vec3 U = p.direction();//U

    /*
    auto a = dot(U, U);//U^2
    auto b = 2.0 * dot(P_C, U);//2(P0-C)U
    auto c = dot(P_C, P_C) - r * r; //((P0-C)^2 - r^2)
    auto discriminant  = b * b - 4 * a * c;
    化  简
    1）向量的平方（点乘自己）等于模的平方
    2）令b=2h,即h=b/2 则可以化简求根公式 得 (-h - √h^2-ac)/a
    */

    //auto a = U.length_squared();
    //auto b_half = dot(P_C, U);
    //auto c = P_C.length_squared() - r * r;
    //auto discriminant = b_half * b_half - a * c;


    //if (discriminant < 0)
    //{
    //    return -1.0;//无解，没有交点
    //}
    //else
    //{
    //    //return (-b - sqrt(discriminant )) / (2.0 * a);//诺有两个交点，则取较小t，即接近射线起点的点
    //    return (-b_half - sqrt(discriminant) / a);
    //}
}

/*
将光线方向缩放为单位长度，根据颜色混合以实现垂直渐变。并利用插值实现颜色的渐变
混合值=(1-t) * 起点值 + t * 终点值 ，t 为射线方向的单位向量缩放到y轴的值
*/
//抗锯齿的处理方法由原来的递归改为循环，以防止栈溢出
__device__ color3 ray_color(const ray& r, hittable & world，curandState *local_rand_state)
{
    ray cur_ray = r;//当前的射线
    color3 cur_attenuation = color3(0.0, 0.0, 0.0);//当前的衰减值
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if (world.hit(r, 0.001, infinity, rec))
        {
            //漫反射
            //point3 target = rec.p + rec.normal + random_unit_hemisphere(rec.normal);//C 为交点外一随机圆的一点
            //return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);//折射光线即为 C-P(交点)

            ray scattered;//散射光
            color3 attenuation;//衰减值
            if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else
            {
                return color3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            //渐变背景,模拟天空效果
            vec3 unit_dirction = unit_vec3(r.direction());
            float t = 0.5f * (unit_dirction.y() + 1.0f);
            return  cur_attenuation * ((1.0f - t) * color3(1.0, 1.0, 1.0) + t * color3(0.5, 0.7, 1.0));
        }
       return color3(0.0, 0.0, 0.0); //超出递归
    }
  
}

//确定图片位置（i，j）,然后计算该位置的最终颜色
__global__ void render(color3* fb, int max_x, int max_y, int samples_per_pixel, camera& cam, hittable& world, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color3 c(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = cam.get_ray(u, v, &local_rand_state);
        c += ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    c /= float(samples_per_pixel);
    c[0] = sqrt(c[0]);
    c[1] = sqrt(c[1]);
    c[2] = sqrt(c[2]);
    fb[pixel_index] = c;
}

//随机态初始化
__global__ void rand_init(curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        curand_init(1984, 0, 0, rand_state);
    }
}
//渲染初始化
__global__ void render_init(int max_x, int max_y, curandState* rand_state) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;

    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}
#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
            new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                        new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main()
{
    // 图片定义
    //生成图片的边框大小 
    int num_pixels_x = 1200;//长
    int num_pixels_y = 800;//宽
    //在GPU上分配, threads_x * threads_y 个区块以进行计算,
    int threads_x = 8;
    int threads_y = 8;
    int samples_per_pixel = 10;
    std::cerr << "Rendering a " << num_pixels_x << "x" << num_pixels_y << " image ";
    std::cerr << "in " << threads_x << "x" << threads_y << " blocks.\n";

    int num_pixels = num_pixels_x * num_pixels_y;
    size_t fb_size = 3 * num_pixels * sizeof(vec3);

    // allocate FB
    //调用 frame buffer，在gpu上计算，然后传送到cpu上
    color3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    //创建2个随机算法状态的对象
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState* d_rand_state2;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    //用两个随机数来初始化世界
    rand_init << <1, 1 >> > (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //世界
    hittable** d_list;
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));

    //相机
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //计时器
    clock_t start, stop;//记时间
    start = clock();

    // Render our buffer
    //渲染FB
    dim3 blocks(num_pixels_x / threads_x + 1, num_pixels_y / threads_y + 1);
    dim3 threads(threads_x, threads_y);
    render << <blocks, threads >> > (fb, num_pixels_x, num_pixels_y,
                                        point3(-2.0, -1.0, -1.0),
                                        vec3(4.0, 0.0, 0.0),
                                        vec3(0.0, 2.0, 0.0),
                                        point3(0.0, 0.0, 0.0));
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


