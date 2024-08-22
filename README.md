# Yet Another Zig Raytracer

## PROGRESS
### CPU zig raytracer: DONE
![image](https://github.com/user-attachments/assets/633597ac-bac3-4161-a8cc-5c87b53b9b08)

### GPU/CUDA zig raytracer: CH 5 DONE (SHADING W NORMALS IMPLEMENTED)
![image](https://github.com/user-attachments/assets/7d035b16-91de-4631-b814-a8b974d72262)
![image](https://github.com/user-attachments/assets/7d749025-92cd-441f-ad42-1decd7609503)

Next up: I need to add hittables, textures, ray sampling, and rays bouncing. Eventually it needs optimization. 0.3s per frame is TRASH

## Helpful Resources
### Raytracer tutorial
Obviously, this bad boy: https://raytracing.github.io/books/RayTracingInOneWeekend.html

This was the guide I went through. It's in C, and I learned a lot by trying to do everything in Zig instead. I feel like if I did it in C I wouldn't have learned as much.


### Zig docs

Zig docs: https://ziglang.org/documentation/master/

Zig.guide: https://zig.guide/language-basics/structs/

By far the most useful, though, was the Zig source code: https://github.com/ziglang/

It's surprisingly readable. It helped me answer questions that I couldn't find info about in either the docs, google, or chatgpt.

If you want to understand how to use certain functions or types or any of that, you can just find what you're working with in here and see all of its code laid out.


### GPU stuff
#### Intro video to GPU stuff
https://www.youtube.com/watch?v=pPStdjuYzSI


#### Zig/cuda example repo
Here's an example of how to use zig as a wrapper for CUDA, so you can use GPU-acceleration in zig:
https://github.com/dnbt777/zig-cuda-utils/

I wrote the cuda kernel with an LLM and used the zig source code to figure out how to link the compiled cuda library in the build.zig.

I am only mid-way through developing the GPU ray tracer, but being able to go back and reference this simple example has been EXTREMELY useful.








## Funny errors
![image](https://github.com/user-attachments/assets/d1bbe2c0-bc01-40e8-b2b2-494ea2019ae3)

cause: spheres have a pointer to a material. I kept overwriting the memory at the pointer. so they all had the same material. lol
```
for (0..n) {
  const mat = [material];
  sphere.smat = &mat;
}
```
I thought it was a buggy random number generator for the longest time
```
// the fix was something like this
for (0..n) {
  sphere.smat = &[material];
}
```

