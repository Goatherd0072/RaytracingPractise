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
const double infinity = std::numeric_limits<double>::infinity(); //正无穷 +∞
const double pi = 3.1415926535897932385; //Π

//功能函数

//角度变弧度
inline double degrees_to_radians(double degrees) 
{
    return degrees * pi / 180.0;
}
inline double random_double() //返回[0,1)的随机数
{
    //rand()返回 0-RAND_MAX ，
    return rand() / (RAND_MAX + 1.0);
}
inline double random_double(double min, double max)//返回[min,max)的随机数
{
    return min + (max - min) * random_double();
}
inline double clamp(double x, double min, double max)//确保值位于范围内
{
    if (x > max) return max;
    if (x < min) return min;
    return x;
}


#endif // !MATH_CONSTANTS_H
