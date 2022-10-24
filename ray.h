#ifndef RAY_H
#define RAY_H

#include "vec3.h"

/*
射线：
参数方程
P(t) = P0 +tU (P0 为射线起点，U 为射线方向，t 为长度
*/

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const point3& Origin, const vec3& Direction) : OriP(Origin), Dir(Direction) {}

	__device__ point3 origin() const { return OriP; }
	__device__ vec3 direction() const { return Dir; }

	//参数方程
	//给定一个t 则可得到向量终点的位置
	__device__ point3 at(double t) const
	{
		return point3(OriP + t * Dir);
	}

public:
	point3 OriP; //射线起点
	vec3 Dir;//射线方向
};

#endif
