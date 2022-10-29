#include <stdio.h>
#include <iostream>

#include"math_constants.h"

#include"color.h"
#include"sphere.h"
#include"hittable_list.h"
#include"camera.h"

#include"vec3.h"
#include"material.h"

using namespace std;

/*
圆的方程： x^2 + y^2 + z^2 = r^2      |   (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2 判断一个点是否在圆上则为两点的距离（目标点和圆心）与半径r的比较
(P-C) * (P-C) = (x-x0)^2 + (y-y0)^2 + (z-z0)^2  P为任意一点 C为圆心(x0,y0,z0)，(P-C)则为点到圆心的向量，代入 射线方程 则为 （P0+tU-C）
判断点是否在圆上即得 （P0+tU-C）^2 = r^2  ① ==> ((P0-C)+tU)^2 = r^2  ②
==> (P0-C)^2 + 2(P0-C)•tU + (tU)^2 - r^2 = 0  >> U^2•t^2 + 2(P0-C)U•t +((P0-C)^2 - r^2)  ③
③方程中 P0为需要判断的点，C为圆心坐标，U为射线方向，r为圆半径，t为未知量。t的不同会使得方程的解不同。
给定任意t与③方程的解的个数即为该射线在圆上的交点个数。
*/
double hit_sphere(const point3& center, double r, const ray& p)
{
    vec3 P_C = p.origin() - center;//(P0-C)
    vec3 U = p.direction();//U

    /* 
    auto a = dot(U, U);//U^2
    auto b = 2.0 * dot(P_C, U);//2(P0-C)U
    auto c = dot(P_C, P_C) - r * r; //((P0-C)^2 - r^2) 
    auto discriminant  = b * b - 4 * a * c;
    化  简
    1）向量的平方（点乘自己）等于模的平方 
    2）令b=2h,即h=b/2 则可以化简求根公式 得 (-h - √h^2-ac)/a
    */

    auto a = U.length_squared();
    auto b_half = dot(P_C, U);
    auto c = P_C.length_squared() - r * r;
    auto discriminant  = b_half * b_half - a * c;

    
    if (discriminant  < 0)
    {
        return -1.0;//无解，没有交点
    }
    else
    {
        //return (-b - sqrt(discriminant )) / (2.0 * a);//诺有两个交点，则取较小t，即接近射线起点的点
        return (-b_half - sqrt(discriminant ) / a);
    }
}

/*
将光线方向缩放为单位长度，根据颜色混合以实现垂直渐变。并利用插值实现颜色的渐变
混合值=(1-t) * 起点值 + t * 终点值 ，t 为射线方向的单位向量缩放到y轴的值
*/
color3 ray_color(const ray& r,const hittable & world, int depth)
{
    hit_record rec;
    
    //限制递归次数
    if (depth < 0)
    {
        return color3(0, 0, 0);
    }

    if (world.hit(r, 0.001, infinity, rec))
    {
        //漫反射
        //point3 target = rec.p + rec.normal + random_unit_hemisphere(rec.normal);//C 为交点外一随机圆的一点
        //return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);//折射光线即为 C-P(交点)

        ray scattered;//散射光
        color3 attenuation;//衰减值
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return color3(0, 0, 0);
    }
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
    auto t = 0.5 * (unit_dirction.y() + 1.0);
    return (1.0 - t) * color3(1.0, 1.0, 1.0) + t * color3(0.5, 0.7, 1.0);
}
hittable_list random_scene() {
    hittable_list world;

    //添加地面材质
    auto ground_material = make_shared<lambertian>(color3(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    //随机生成小球
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color3::random() * color3::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color3::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }
    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color3(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color3(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

int main()
{
    // 图片定义
    const auto aspect_ratio = 3.0/2.0;//屏幕比例
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    //const int image_width = 256;  // columns
    //const int image_height = 256; // rows
    const int samples_per_pixel = 500;
    const int max_depth = 50;//采样次数

    //世界
    //hittable_list world;
    auto world = random_scene();

    //相机
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // 渲染
    cout << "P3\n"
         << image_width << " " << image_height << "\n255\n";

    for (int i = image_height - 1; i >= 0; i--)//每行
    {
        cerr << "RemainLine: " << i << " " << flush;
        for (int j = 0; j < image_width; j++)//每列
        {
            color3 pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; s++)
            {
                auto u = (j + random_double()) / (image_width - 1);
                auto v = (i + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);

                pixel_color += ray_color(r, world, max_depth);
            }
            Print_Color(cout, pixel_color, samples_per_pixel);
        }
    }
    cerr << "\nDone" << endl;

  
}
