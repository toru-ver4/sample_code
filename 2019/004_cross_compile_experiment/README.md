# クロスコンパイルで Windows 用のバイナリを作る


## 事前準備

まず WSL 上に Ubuntu 18.04 を用意する。Windows Store から普通にインストールできる。

続いて、必要と思われるパッケージを入れる（```build-essential``` は要らなかったかも😅）

```Shell
sudo apt-get update
sudo apt-get install build-essential mingw-w64 -y
```

## コード作成

```C++:main.cpp
#include <iostream>

int main()
{
    std::cout << "Hello, World" << std::endl;
}
```

## コンパイル

```Shell
x86_64-w64-mingw32-g++ main.cpp -o main.exe -static-libstdc++ -static-libgcc
```

## 実行

```Batchfile
C:\home\sample_code\2019\004_cross_compile_experiment>main.exe
Hello, World
```

## 感想

あれ、普通に動いたぞ。会社でハマってたのは一体何だったんだ…。

## 参考資料

kakurasanのLinux備忘録, "Debian/Ubuntuでmingw-w64を用いてWindows向けのプログラムをコンパイルする", https://kakurasan.blogspot.com/2015/07/debianubuntu-mingw-crosscompile.html
