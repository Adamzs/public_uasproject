cmake_minimum_required(VERSION 3.6)
project(collect_images)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    colorspaces.c
    colorspaces.h
    ConvertColor.cpp
    ConvertColor.h
    format_converter.hpp
    jpeg_decoder.c
    jpeg_decoder.h
    main.cpp
        Track.cpp
    Track.hpp
    withrobot_camera.cpp
    withrobot_camera.hpp
    withrobot_debug_print.h
    withrobot_utility.cpp
    withrobot_utility.hpp)

add_executable(collect_images ${SOURCE_FILES})