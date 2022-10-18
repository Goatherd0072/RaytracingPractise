#ifndef  SPHERE_H
#define SPHERE_H

#include"hittable.h"
#include"vec3.h"

class sphere : public hittable
{
public:
	sphere(){}
    sphere(point3 c, double r,shared_ptr<material> m) :center(c), radius(r), mat_ptr(m) {};

	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& record) const override;

public:
	point3 center;
	double radius;
    shared_ptr<material> mat_ptr;

};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
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
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}

#endif // ! SPHERE_H
