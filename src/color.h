#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

#include<iostream>


void Print_Color(ostream & out,color3 Color_Pixel,int samples_per_pixel)
{
	auto r = Color_Pixel.x();
	auto g = Color_Pixel.y();
	auto b = Color_Pixel.z();

	//处理颜色与采样次数
	//将颜色值除以采样次数，并取其平方根以进行 gamma矫正
	auto scale = 1.0 / samples_per_pixel;
	r = sqrt(r * scale);
	g = sqrt(g * scale);
	b = sqrt(b * scale);

	//将颜色转化为255标准
	out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << " "
		<< static_cast<int>(256 * clamp(g, 0.0, 0.999)) << " "
		<< static_cast<int>(256 * clamp(b, 0.0, 0.999)) << endl;
}

#endif