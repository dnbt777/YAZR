clear;nvcc -shared -o ./cudalibs/libutils.so -Xcompiler -fPIC ./cudalibs/utils.cu -lglut -lglfw -lGL -lGLU;zig build
