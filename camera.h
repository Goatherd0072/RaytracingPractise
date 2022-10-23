#ifndef CAMERA_H
#define CAMERA_H

#include "math_constants.h"

//焦距 focal length， 该项目中可以理解为，投影点到像平面之间的距离
//对焦距离 focus distance， 焦点到焦平面（focal plane）的距离，在focal plane上的点，在感光元件上都有“perfectly sharp”的点与它对应，当物体越靠近Focal Plane，得到的图像越清晰。

class camera
{
public:
	camera( point3 lookfrom,//起点
			point3 lookat,//终点
			vec3 Vup,//向上的向量
			float vFOV, //垂直的FOV
			float aspect_ratio,//屏幕比例
			float aperture,//光圈直径
			float focus_dist //对焦距离 
			//在透镜前设置光圈和对焦距离，实现散焦模糊（景深）
	){
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

		w = unit_vec3(lookfrom - lookat);
		u = unit_vec3(cross(Vup, w));
		v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;//水平方向 x轴
		vectical = focus_dist * viewport_height * v;//垂直方向 y轴
		lower_left_corner = origin - horizontal / 2 - vectical / 2 - focus_dist * w;//左下角 视窗起点

		lens_radius = aperture / 2;
	 }
	ray get_ray(float s,float t) const//从相机中心生成射线
	{
		//Defocus Blur 失焦模糊（景深）
		vec3 rd = lens_radius * random_in_uint_disk();//在透镜内产生随机点
		vec3 offset = u * rd.x() + v * rd.y();//在u,v两个方向上添加偏移量

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vectical - origin - offset);
	}
	
private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vectical;
	vec3 u, v, w;
	float lens_radius;//透镜半径
};



#endif // !CAMERA_H
