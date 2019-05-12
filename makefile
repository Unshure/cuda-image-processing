CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config --cflags --libs opencv`
LIBS = $(OPENCV)

all: process

clean:
	rm -rf *.o process

process: cudaProcess.o
	$(CC) $(CFLAGS)  process.cpp cudaProcess.o -o process $(LIBS) -L/usr/lib/cuda/lib64 -lcuda -lcudart

cudaProcess.o:
	nvcc -c -arch=sm_20 cudaProcess.cu
