CC = x86_64-w64-mingw32-g++
CFLAGS = -Wall -static-libstdc++ -static-libgcc -static
INCLUDE = -I./include
LIB = -L./lib

ALL: device_control.o test.o main.o
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) -o ./bin/subprocess.exe ./bin/device_control.o ./bin/test_func.o ./bin/main.o

main.o: ./src/main.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) -o ./bin/main.o -c ./src/main.cpp

test.o: ./src/test_func.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) -o ./bin/test_func.o -c ./src/test_func.cpp

device_control.o: ./src/device_control.cpp
	$(CC) $(CFLAGS) $(INCLUDE) $(LIB) -o ./bin/device_control.o -c ./src/device_control.cpp

clean:
	rm ./bin/*.o
	rm ./bin/*.exe
