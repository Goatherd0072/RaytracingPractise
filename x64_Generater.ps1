.\x64\Debug\RaytracingPractise.exe > image1.ppm
type .\image1.ppm | Out-File -Encoding "Default" .\image.ppm
magick image.ppm image.png
cp .\image.ppm .\image\image.ppm
cp .\image.png .\image\image.png
rm .\image.png
rm .\image.ppm
rm .\image1.ppm