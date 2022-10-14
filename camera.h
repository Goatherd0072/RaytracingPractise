#ifndef CAMERA_H
#define CAMERA_H

#include "math_constants.h"

class camera
{
public:
	camera()
	{
		auto aspect_ratio = 16.0 / 9.0;//屏幕比例
		auto viewport_height = 2.0;
		auto viewport_width = aspect_ratio * viewport_height;
		auto focal_length = 1.0;//焦距

		origin = point3(0, 0, 0);
		horizontal = vec3(viewport_width, 0, 0);//水平方向 x轴
		vectical = vec3(0, viewport_height, 0);//垂直方向 y轴
		lower_left_corner = origin - horizontal / 2 - vectical / 2 - vec3(0, 0, focal_length);//左下角 视窗起点
	 }
	ray get_ray(double u,double v) const//从相机中心生成射线
	{
		return ray(origin, lower_left_corner + u * horizontal + v * vectical - origin);
	}
	

public:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vectical;
};



#endif // !CAMERA_H
