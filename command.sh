#clear;nvcc -shared -o ./cudalibs/libutils.so -Xcompiler -fPIC ./cudalibs/utils.cu;zig build
clear;nvcc -shared -Xcompiler -fPIC -o ./cudalibs/libutils.so ./cudalibs/utils.cu -lGL;zig build
