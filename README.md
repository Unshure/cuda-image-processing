# cuda-image-processing
This is a set of image processing tools implemented in regular C++ and with Cuda. These tools include:
- Grayscale
- Blur
- Line Detection

--------------------------------------------------------------------------
## Compiling the project
This project uses the following libraries and has compiled with the versions shown:
-Opencv (2.4.5), (3.2.0), (3.3.1)
-Cuda (9.1.85)

Move into the main directory of this project and run the "make" command. A file nammed "process" should appear afterwards.

If your computer does not have a gpu, you can still compile this project, it will just not have the gpu tools. In order to do this, run the command "make cpu" instead.

----------------------------------------
## Running the project
The execution of this project is done with the following command and flags:

#### ./process "path/to/image" (-cuda) (-g | -b # | -l)

- -cuda: Optional flag to tells the program if it should use the regular c++ or the cuda version of the tool
- -g: Grayscales the image
- -b #: Blurs the image where # is some number >0 to determine the strength of the blur
- -l: Transforms an image to highlight the strong lines in an image
