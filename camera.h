#ifndef CAMERA_H
#define CAMERA_H

#include "math_constants.h"

class camera
{
public:
	camera( point3 lookfrom,//起点
			point3 lookat,//终点
			vec3 Vup,//向上的向量
			double vFOV, //垂直的FOV
			double aspect_ratio)
	{
		//垂直方向的 FOV
		auto theta = degrees_to_radians(vFOV);
		auto h = tan(theta / 2);
		auto viewport_height = 2.0 * h;
		auto viewport_width = aspect_ratio * viewport_height;

		////水平方向的 FOV		
		//auto theta = degrees_to_radians(hFOV);
		//auto h = tan(theta / 2);
		//auto viewport_width = 2.0 * h;
		//auto viewport_height = aspect_ratio * viewport_width;

		//auto focal_length = 1.0;//焦距

		auto w = unit_vec3(lookfrom - lookat);
		auto u = unit_vec3(cross(Vup, w));
		auto v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;//水平方向 x轴
		vectical = viewport_height * v;//垂直方向 y轴
		lower_left_corner = origin - horizontal / 2 - vectical / 2 - w;//左下角 视窗起点
	 }
	ray get_ray(double s,double t) const//从相机中心生成射线
	{
		return ray(origin, lower_left_corner + s * horizontal + t * vectical - origin);
	}
	
private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vectical;
};



#endif // !CAMERA_H
