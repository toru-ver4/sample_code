cmake_minimum_required(VERSION 3.20)
project(my_capture_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(my_capture_app main.cpp)

# WindowsApp.lib は C++/WinRT の WinRT 활성化に必要な定番リンク
target_link_libraries(my_capture_app
  PRIVATE
    windowsapp
    d3d11
    dxgi
    ole32
    runtimeobject
)
