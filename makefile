CC=g++

CFLAGS=-g -Wall -std=c++11

OPENCV = `pkg-config --cflags --libs opencv`
LIBS = $(OPENCV)

all: process

clean:
	rm -rf *.o process

process: cudaProcess.o
	$(CC) -o process $(LIBS) -L/path/to/my/openCV/lib -lopencv_core -lopencv_imgcodecs -L/usr/lib/cuda/lib64 -lcuda -lcudart process.cpp cudaProcess.o

cudaProcess.o:
	nvcc -c -arch=sm_20 cudaProcess.cu
