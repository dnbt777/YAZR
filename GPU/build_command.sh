# clear;nvcc -shared -o ./cudalibs/libutils.so -Xcompiler -fPIC ./cudalibs/utils.cu -lglut -lglfw -lGL -lGLU;zig build
nvcc -shared -o ./cudalibs/libutils.so -Xcompiler -fPIC ./cudalibs/utils.cu -lglut -lglfw -lGL -lGLU -lGLEW;zig build
