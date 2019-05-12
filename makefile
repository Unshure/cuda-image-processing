CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -L/usr/lib/cuda/lib64 -lcuda -lcudart

all: process

clean:
	rm -rf *.o process

process: cudaProcess.o
	$(CC) $(CFLAGS) -o process $(LIBS) process.cpp cudaProcess.o

cudaProcess.o:
	nvcc -c -arch=sm_70 cudaProcess.cu
