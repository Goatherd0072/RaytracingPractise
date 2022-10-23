#ifndef MATH_CONSTANTS_H
#define MATH_CONSTANTS_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

//#include "vec3.h"
//#include "ray.h"

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

//常量
const float infinity = std::numeric_limits<float>::infinity(); //正无穷 +∞
const float pi = 3.1415926535897932385; //Π

//功能函数

//角度变弧度
inline float degrees_to_radians(float degrees) 
{
    return degrees * pi / 180.0;
}
inline float random_float() //返回[0,1)的随机数
{
    //rand()返回 0-RAND_MAX ，
    return rand() / (RAND_MAX + 1.0);
}
inline float random_float(float min, float max)//返回[min,max)的随机数
{
    return min + (max - min) * random_float();
}
inline float clamp(float x, float min, float max)//确保值位于范围内
{
    if (x > max) return max;
    if (x < min) return min;
    return x;
}


#endif // !MATH_CONSTANTS_H
