#ifndef SPHERE_CUH
#define SPHERE_CUH

#include"hittable.cuh"
#include"vec3.cuh"

class sphere : public hittable
{
public:
    __device__ sphere(){}
    __device__ sphere(point3 c, float r,shared_ptr<material> m) :center(c), radius(r), mat_ptr(m) {};

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;

public:
	point3 center;
	float radius;
    shared_ptr<material> mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    vec3 P_C = r.origin() - center;//(P0-C)
    vec3 U = r.direction();//U

    auto a = U.length_squared();
    auto b_half = dot(P_C, U);
    auto c = P_C.length_squared() - radius * radius;
    auto discriminant  = b_half * b_half - a * c;

    if (discriminant  < 0) return false; //未相交
    auto sqrtD = sqrt(discriminant);

    //判断两根是否在所需范围内
    auto root = (-b_half - sqrtD) / a;
    if (root<t_min || root>t_max)
    {
        root = (-b_half + sqrtD) / a;
        if (root<t_min || root>t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius; //法线向量单位化
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif // ! SPHERE_H
