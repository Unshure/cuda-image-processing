CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config --cflags --libs opencv`
LIBS = $(OPENCV)

all: process

clean:
	rm -rf *.o process

process: cudaProcess.o
	$(CC) $(CFLAGS) $(LIBS) -lcudart process.cpp -o process 

cudaProcess.o:
	nvcc -c -arch=sm_20 cudaProcess.cu
