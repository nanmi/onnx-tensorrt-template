<!--
 * @Description: OpenCV-GPU TensorRT
 * @Author: nanmi
 * @Date: 2021-06-29 09:16:35
 * @LastEditTime: 2021-06-29 16:03:14
 * @LastEditors: nanmi
 * @GitHub:github.com/nanmi
 -->

# ONNX TensorRT Template
This project base on [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt)


# News

It can speed up the whole pipeline on GPU, greatly improve the operation efficiency, and customize the pre-processing and post-processing on GPU - 2021-6-29


# Features
- [x] Preprocess in GPU
- [x] Postprocess in GPU
- [x] run whole pipeline in GPU easily
- [x] Custom onnx model output node
- [x] Engine serialization and deserialization auto
- [x] INT8 support

# System Requirements
cuda 10.0+

TensorRT 7

OpenCV 4.0+ (build with opencv-contrib module)

# Installation
Make sure you had install dependencies list above
```bash
# clone project and submodule
git clone {this repo}

cd {this repo}

mkdir build && cd build && cmake .. && make
```
Then you can intergrate it into your own project with libtinytrt.so and Trt.h

# Docs

example cxx code for how to use opencv gpu version in TensorRT inference.

# About License

For the 3rd-party module and TensorRT, you need to follow their license

For the part I wrote, you can do anything you want

