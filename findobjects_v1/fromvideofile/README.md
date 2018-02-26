# The Example of using the Withrobot oCam API to get frames from the camera and then process them using with OpenCV. (Linux only.)
This example shows how to get image from the oCam using the Withrobot camera API. And also shows how to control the oCam using the Withrobot camera API. Please refer to the comments in `main.cpp` for details.

## Sample code configuration
- main.cpp : example source file
- withrobot_camera.hpp : withrobot camera API header file
- withrobot_camera.cpp : withrobot camera API source file
- withrobot_utility.hpp: withrobot utility API source file
- Makefile : The Makefile for build this example.
- README.md : This file.

## How to build on linux
Requirements
- libv4l       (video for linux Two)
- libudev       (udev, the device manager for the Linux kernel)
- libopencv-dev (development files for OpenCV)

1.Download and install from linux package manager(e.g. apt).
```
$ sudo apt-get install libv4l-dev libudev-dev libopencv-dev
```

2.Build.
```
$ cd ./withrobotcamera
$ gcc -c -o jpeg_decoder.o jpeg_decoder.c 
$ gcc -c -o colorspaces.o colorspaces.c 
$ make all
```

## How to run
```
$ ./build/bin/withrobotcamera
```


When using the oCam camera the image format is YUYV so switch the format_converter line appropriately
When using the SeeCam camera the image format is UYVY so switch the format_converter line appropriately



