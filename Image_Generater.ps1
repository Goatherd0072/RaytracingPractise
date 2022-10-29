try {
    $filename = Read-Host "input image name"
    $filenamePNG = $filename + ".png"
    $filenamePPM = $filename + ".ppm"
    $filename1PPM = $filename + "1" + ".ppm"

    .\RaytracingPractise.exe > $filename1PPM 
    type .\$filename1PPM  | Out-File -Encoding "Default" .\$filenamePPM 
    magick $filenamePPM  $filenamePNG
    rm .\$filename1PPM
    echo "Finish!"
}
catch {
    echo "Something Wrong!"
}