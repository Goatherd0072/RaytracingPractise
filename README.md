# RaytracingPractise

一个根据 [book: Ray Tracing in One Weekend](https://github.com/RayTracing/raytracing.github.io) 的小练习

# 使用

将ppm文件转为png格式需要用到[imagemagick](https://imagemagick.org/script/download.php)

编译[./src文件夹](./src/)下的代码后，将 [Image_Generater.ps1](./Image_Generater.ps1) 放入.exe文件目录下，然后运行

# 加速

因为代码默认执行用的CPU，导致生成速度巨慢，可以使用CUDA加速生成，参考了Roger Allen在NVIDIA的[这篇博客](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
