cd C:\home\build_tools ^ 
git clone --branch 31.0.3 --recursive https://github.com/obsproject/obs-studio.git ^ 
cd C:\home\build_tools\obs-studio ^ 
rmdir /s /q .git ^ 
rmdir /s /q .github ^ 
cd plugins ^ 
git clone --branch 2.4.3 https://github.com/exeldro/obs-shaderfilter.git ^ 
echo add_obs_plugin(obs-shaderfilter PLATFORMS WINDOWS) >> "CMakeLists.txt"
