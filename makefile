CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

HEADER=include/process.hpp

all: process

clean:
	rm -rf *.o process

process:
	$(CC) $(CFLAGS) process.cpp -o process $(LIBS) gpu

gpu:
	nvcc -c -arch=sm_75 cudaProcess.cu
