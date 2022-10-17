try {
    $filename = Read-Host "input image name"
    $filenamePNG = $filename + ".png"
    $filenamePPM = $filename + ".ppm"
    $filename1PPM = $filename + "1" + ".ppm"

    .\x64\Debug\RaytracingPractise.exe > $filename1PPM 
    type .\$filename1PPM  | Out-File -Encoding "Default" .\$filenamePPM 
    magick $filenamePPM  $filenamePNG
    cp .\$filenamePPM .\image\$filenamePPM
    cp .\$filenamePNG .\image\$filenamePNG
    rm .\$filenamePNG
    rm .\$filenamePPM
    rm .\$filename1PPM
    echo "Finish!"
}
catch {
    echo "Something wrong!"
}