#!/bin/sh

cp ./img/prime_on_lab_original.png ./img/mgif/eq05_00.png
cp ./img/prime_on_lab_original.png ./img/mgif/eq08_00.png
cp ./img/prime_on_lab_original.png ./img/mgif/eq058_00.png

cp "./img/prime_on_lab_Apply X'_D65.png" ./img/mgif/eq05_01.png
cp "./img/prime_on_lab_Apply X'_D65.png" ./img/mgif/eq58_00.png
cp "./img/prime_on_lab_Apply X'_D65.png" ./img/mgif/eq058_01.png

cp "./img/prime_on_lab_Apply X'_D65 and Y'_D65.png" ./img/mgif/eq08_01.png
cp "./img/prime_on_lab_Apply X'_D65 and Y'_D65.png" ./img/mgif/eq58_01.png
cp "./img/prime_on_lab_Apply X'_D65 and Y'_D65.png" ./img/mgif/eq058_02.png

ffmpeg -r 2 -i ./img/mgif/eq05_%02d.png -filter_complex "split[a][b];[a]palettegen=stats_mode=single[pal];[b][pal]paletteuse=new=1" ./img/mgif/agif_eq05.gif -y
ffmpeg -r 2 -i ./img/mgif/eq08_%02d.png -filter_complex "split[a][b];[a]palettegen=stats_mode=single[pal];[b][pal]paletteuse=new=1" ./img/mgif/agif_eq08.gif -y
ffmpeg -r 2 -i ./img/mgif/eq58_%02d.png -filter_complex "split[a][b];[a]palettegen=stats_mode=single[pal];[b][pal]paletteuse=new=1" ./img/mgif/agif_eq58.gif -y
ffmpeg -r 2 -i ./img/mgif/eq058_%02d.png -filter_complex "split[a][b];[a]palettegen=stats_mode=single[pal];[b][pal]paletteuse=new=1" ./img/mgif/agif_eq058.gif -y
