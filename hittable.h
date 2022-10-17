#ifndef HITTABLE_H
#define HITTABLE_H

#include"ray.h"
#include "math_constants.h"

class material;

//储存光线交互的有效值
struct hit_record
{
	point3 p;//交点
	vec3 normal;//法线
	double t;//距离
	bool front;//外面
	shared_ptr<material> mat_ptr;//程序开始时候，光线碰撞到表面时，则会将此指针指向球类里的hit_record指针

	//判断射线位于物体内部还是外部，并将法线调整未向外的法线(光线方向与法线方向一样，则在内部。相反则在外部)
	inline void set_front_normal(const ray& r, const vec3& outward_normal)
	{
		front = dot(r.direction(), outward_normal) < 0;//小于0则相反，在外
		normal = front ? outward_normal : -outward_normal;
	}
};

class hittable
{
public:
	//当t位于区间（t_min，t_max）内时，则记录下光线击中点的信息。 //用纯虚函数，方便后面不同的继承类更具自身属性更改
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& record) const = 0;
};

#endif // !HITTABLE_H
