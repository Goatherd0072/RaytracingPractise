#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>


using namespace std;

class vec3
{
public:
	//初始化构造函数
    vec3() : e{ 0, 0, 0 } {};
    vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {};//x,y,z
	
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }
	
	/* 
	double &operator[](int i); 
	double operator[](int i)const;
	要操作数组中的元素当然是第一个；要给一个变量赋值。就是第二个了。
	一个用于左值，一个用于右值。

	a[3] = 5; 这里用的是double & operator[](int i);
	double x = a[3]; 这里用的是double operator[](int i)const;
	*/
	double operator[](int i) const { return e[i]; }
	double & operator[](int i) { return e[i]; }
	
    //相反向量
    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

	//向量加法
    vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
	
	//向量的标量乘法
	vec3& operator*=(const double& t)
	{
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}
	
    //向量的标量除法(等于乘以倒数)
	vec3& operator/=(const double& t)
	{
		return *this *= 1 / t;
	}
	
	//向量的模的平方
	double length_squared() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}
	//向量的模
	double length() const
	{
		return sqrt(length_squared());
	}
	//返回随机的向量
	inline static vec3 random()
	{
		return vec3(random_double(), random_double(), random_double());
	}
	inline static vec3 random(double min, double max)
	{
		return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
	}
	//判断向量是否在各个方向都为0
	bool near_zero() const
	{
		const auto s = 1e-8;
		return fabs(e[0]) < s && fabs(e[1]) < s && fabs(e[2]) < s;
	}
	
public:
    double e[3]; //坐标
};

//别名声明， 使坐标点和颜色和向量用同一个类
using point3 = vec3;
using color3 = vec3;

/*
在类内声明类外定义的时候，会报错operator 此运算符函数的参数太多;这是因为在类内定义的话会默认有一个this指针，因此就超过了两个参数。
应该把operator函数在类外做为一个全局函数
*/

//vec3类的功能函数

//输出流
inline ostream& operator<<(ostream& out, const vec3& v)
{
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
//加法
inline vec3 operator +(const vec3 & e1,const vec3 & e2)
{
	return vec3(e1.e[0] + e2.e[0], e1.e[1] + e2.e[1], e1.e[2] + e2.e[2]);
}
//减法
inline vec3 operator -(const vec3& e1, const vec3& e2)
{
	return vec3(e1.e[0] - e2.e[0], e1.e[1] - e2.e[1], e1.e[2] - e2.e[2]);
}
//向量间乘法
inline vec3 operator *(const vec3& e1, const vec3& e2)
{
	return vec3(e1.e[0] * e2.e[0], e1.e[1] * e2.e[1], e1.e[2] * e2.e[2]);
}
//标量乘法
//v * t
inline vec3 operator *(const vec3& v, const double& t)
{
	return vec3(v.e[0]*t, v.e[1]*t,v.e[2]*t);
}
// t * v
inline vec3 operator *(const double t,const vec3 & v)
{
	return v * t;
}
//向量除法
inline vec3 operator /(vec3 v, double t)
{
	return v*(1/t);
}
//点乘（数量积）标量  a·b=a1​b1​+a2​b2​+…+an​bn
inline double dot(const vec3 & v1, const vec3 & v2)
{
	return (v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2]);
}
//                         x1 y1 z1
//叉乘（向量积）矢量   aXb = x2 y2 z2  =(y1*z2-z1*y2, z1*x2-x1*z2, x1*y2-y1*x2 )
//
inline vec3 cross(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[1]*v2.e[2]-v1.e[2]*v2.e[1],
				v1.e[2]*v2.e[0]-v1.e[0]*v2.e[2],
				v1.e[0]*v2.e[1]-v1.e[1]*v2.e[0]);
}
//单位向量 = a/|a|
inline vec3 unit_vec3(vec3 v)
{
	return v / v.length();
}

//随机生成碰撞点漫反射范围的随机单位球
vec3 random_unit_sphere()
{
	while (true)
	{
		auto c = vec3::random(-1, 1);
		if (c.length_squared() >= 1) continue;//保证球的大小在单位球内
		return c;
	}
}
//碰撞点与单位球的随机向量，以实现 Lambertian反射
vec3 random_unit_vector()
{
	return unit_vec3(random_unit_sphere());
}
//随机生成碰撞点漫反射范围的随机单位半球,以便实现半球散射
vec3 random_unit_hemisphere(const vec3& normal)
{
	vec3  unit_sphere = random_unit_sphere();
	if (dot(unit_sphere, normal) > 0.0)//如果半球和法线在一个表面
	{
		return unit_sphere;
	}
	else
	{
		return -unit_sphere;
	}
}

//计算反射光向量，v入射光线，n法线
vec3 reflect(const vec3& v,const vec3& n)
{
	return v - 2 * dot(v, n) * n; //放射光线理应为v+2b， 但dot(v, n)为b的长度，由于入射光线和法线夹角大于90，所以2 * dot(v, n) * n为负值
}

//计算折射光向量 R为入射光的方向向量 n为入射平面的法线，ratio 为两个介质折射率之比 n1/n2
//将折射光线向量分解为 垂直于法线的 R1 和平行于法线的 R2
//cosθ = (-R * n) (由 a⋅b=|a||b|cosθ 当a、b为单位向量时推导)
//R1 = η/η′ * (R+cosθn) = η/η′ * (R+(−R⋅n)n)
//R2 = -sqrt(1 - |R1|^2 )* n
//详细推导过程 ： https://zhuanlan.zhihu.com/p/91129191  https://github.com/RayTracing/raytracing.github.io/issues/1082 TODO：
vec3 refract(const vec3& R, const vec3& n, double ratio)
{
	auto cos_theta = fmin(dot(-R, n), 1.0);
	vec3 R1 = ratio * (R + cos_theta * n);
	vec3 R2 = -sqrt(fabs(1.0 - R1.length_squared())) * n;
	return R1 + R2;
}

#endif