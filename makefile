CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all: process

clean:
	rm -rf *.o process

process: cudaProcess.o
	$(CC) $(CFLAGS) process.cpp -o process $(LIBS) cudaProcess.o

cudaProcess.o:
	nvcc -c -arch=sm_70 cudaProcess.cu
