CC=nvcc
CCCpu = g++

CPUFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

CFLAGS=-std=c++11 -x cu

HEADER=include/process.h

default: process

clean:
	rm process cudaProcess

image:
	$(CC) $(CFLAGS) process.cu -arch=compute_35 -o cudaProcess
cpu:
	$(CCCpu) $(CPUFLAGS) process.cpp -o process $(LIBS)
