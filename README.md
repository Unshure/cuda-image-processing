# cuda-image-processing
This is a set of image processing tools implemented in regular C++ and with Cuda. These tools include:
- Grayscale
- Blur
- Line Detection

--------------------------------------------------------------------------
## Compiling the project
Move into the main directory of this project and run the "make" command. A file nammed "process" should appear afterwards.

----------------------------------------
## Running the project
The execution of this project is done with the following command and flags:

#### ./process "path/to/image" (-cuda) (-g | -b # | -l)

- -cuda: Optional flag to tells the program if it should use the regular c++ or the cuda version of the tool
- -g: Grayscales the image
- -b #: Blurs the image where # is some number >0 to determine the strength of the blur
- -l: Transforms an image to highlight the strong lines in an image
