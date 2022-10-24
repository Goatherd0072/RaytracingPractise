#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "ray.cuh"
#include "hittable.cuh"

//将需要的参数集中与一个结构体中，以便类与类直接的交换
//
struct hit_record;

class material
{
public:
	//判断是否为散射光
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
									color3& attenuation, ray& scattered, 
									curandState* local_rand_state) const = 0;
};

/* albedo为反照率，绝对黑体（black body）的反照率是0。煤炭呈黑色，反照率 接近0，因为它吸收了投射到其表面上的几乎所有可见光。
镜面将可见光几乎全部反射出去，其反照率接近1。
与 reflectance（反射率）是有区别的。反射率用来表示单一一种波长的反射能量与入射能量之比；而反照率用来表示全波段的反射能量与入射能量之比 
fuzz为模糊值，在反射光方向生成一个半径为fuzz的圆，并在圆内随机生成一点，使反射光指向该点，以实现模糊效果 
 */

//散射光
class lambertian : public material
{
public:
	__device__ lambertian(const color3& a) : albedo(a){}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
									color3& attenuation, ray& scattered,
									curandState* local_rand_state)
	const override
	{
		auto scattered_dirction = rec.normal + random_unit_vector(local_rand_state);
		//防止随机生产一个和法线向量相反的向量，导致散射光线的向量为0
		if (scattered_dirction.near_zero()) 
		{
			scattered_dirction= rec.normal;
		}

		scattered = ray(rec.p, scattered_dirction);
		attenuation = albedo;

		return true;
	}
public:
	color3 albedo;
	//float p;
	//color3 aldedo_p = albedo / p;//也可以以某个概率p进行散射
};

//镜面反射
class metal : public material
{
public:
	__device__ metal(const color3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
									color3& attenuation, ray& scattered,
									curandState* local_rand_state)
	const override
	{
		vec3 reflect_d = reflect(unit_vec3(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflect_d + fuzz * random_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal)) > 0; //判断散射方向和法线方向
	}

public:
	color3 albedo;
	float fuzz;//模糊值，
};

//Dielectrics 介质，模拟光传播过程中的介质，以实现玻璃、钻石等折射的效果
// 当此介质的物体，半径为负数时，不会影响其几何形状，但由于法线向内，所以可以用作气泡来制作空心玻璃球
class dielectric : public material
{
public:
	__device__ dielectric(float IR) : ir(IR){}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
									color3& attenuation, ray& scattered,
									curandState* local_rand_state)	
	const override
	{
		attenuation = color3(1.0, 1.0, 1.0);
		float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;//如果在外侧，根据折射定律，折射率则为倒数

		vec3 unit_dir = unit_vec3(r_in.direction());
		//vec3 unit_n = unit_vec3(rec.normal);

		float cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0);
		float sin_theta = sqrt(1 - cos_theta * cos_theta);

		bool can_Refraction = sin_theta * refraction_ratio > 1.0;//判断能不能进行折射
		vec3 direction;

		if (can_Refraction || Schlick_Approximation(cos_theta, refraction_ratio) > random_float(local_rand_state))
		{
			//必须进行反射
			direction = reflect(unit_dir, rec.normal);
		}
		else
		{
			//可以进行折射
			direction = refract(unit_dir, rec.normal, refraction_ratio);
		}

		scattered = ray(rec.p, direction);
		return true;
	}

public:
	float ir;//折射率

private:
	//使用Schlick Approximation来模拟玻璃等随角度进行变化的折射率。 https://en.wikipedia.org/wiki/Schlick%27s_approximation 
	// 折射率 r = R0 + (1 - R0) ( (1 - cos) )^5 
	// R0 = ( (n1 - n2)/(n1 + n2) )^2 
	//R0是光的反射系数的平行于法线， n1 n2为两侧介质的折射率， 由于图形学入射光线大部分情况都在空气，所以 n1 近似于 1
	__device__ static float Schlick_Approximation(float cos_t, float index_ref)
	{
		auto R0 = (1 - index_ref) / (1 + index_ref);
		R0 *= R0;
		return R0 + (1 - R0) * pow((1 - cos_t), 5);
	}
	
};

#endif 