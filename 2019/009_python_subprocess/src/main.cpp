#include <iostream>
#include "test_func.hpp"
#include "device_control.hpp"


int main()
{
    std::cout << "Hello, World" << std::endl;
    int y = hoge_func(3);
    printf("%d\n", y);

    // serach device

    // wait command
    device_control_main();
}
