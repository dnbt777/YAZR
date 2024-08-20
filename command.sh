clear;nvcc -shared -o ./cudalibs/libutils.so -Xcompiler -fPIC ./cudalibs/utils.cu;zig build
