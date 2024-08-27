#include <cuda_runtime.h>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <cuda_gl_interop.h>

__managed__ int WIDTH;
__managed__ int HEIGHT;
__managed__ int CHANNELS;
__managed__ GLuint textureID;
__managed__ GLenum target;
__managed__ cudaGraphicsResource* cudaResource;
__managed__ unsigned char* d_image;

__device__ float magnitude(float var[3]) {
    return sqrt(var[0]*var[0] + var[1]*var[1] + var[2]*var[2]);
}

__device__ void unitvector(float var[3]) {
    float mag = magnitude(var);

    if (mag != 0.0) {
        var[0] /= mag;
        var[1] /= mag;
        var[2] /= mag;
    } else {
        var[0] = 0.0;
        var[1] = 0.0;
        var[2] = 0.0;
    }
}

extern "C" int cuda_device_check() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return -1; // Exit if no device is found
    }

    // Print device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Get properties of the first device
    printf("Using device: %s\n", deviceProp.name);

    return 0;
}
// nah opengl now
// __managed__ float d_R[1000 * 1000], d_G[1000*1000], d_B[1000*1000];


// todo - make *tex managed
__global__ void renderKernel(
        unsigned char* screen_tex, int width, int height,
        const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
        const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
        const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
        const float origin0, const float origin1, const float origin2,
        int samples
        ) {
        // int samples_per_pixel, int depth,
        // int hittables_flattened, int num_hittables) { //idk what to do about hittable type
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pixel_center[3];
    float color[3]; 
    float direction[3];

    float unit_direction[3];
    float magnitude;

    // execute threads from (0,0) to (height*ceil(sqrt(samples_per_pixel)), width*ceil(sqrt(samples_per_pixel)))
    // space has to fit each thread or whatever
    // int sqrt_samples_per_pixel = ceil(sqrt(samples_per_pixel)); // may cause issues affecting memory in outside executed threads...
    // one step at a time. first we just trace a single ray per pixel
    for (int i=0;i<samples;i++) {
        if (row < height && col < width) {
            int idx = (row*width + col) * 3; // RGB - `texture` is [R, G, B, R, G, B, R, G, ...]
            // // // // // // // ray color func here // // // // // // //
            // pixel's center in the viewport
            // uh, I think...
            pixel_center[0] = pixel00_loc0 + (row * pixel_delta_u0) + (col * pixel_delta_v0);
            pixel_center[1] = pixel00_loc1 + (row * pixel_delta_u1) + (col * pixel_delta_v1);
            pixel_center[2] = pixel00_loc2 + (row * pixel_delta_u2) + (col * pixel_delta_v2);

            direction[0] = pixel_center[0] - origin0;
            direction[1] = pixel_center[1] - origin1;
            direction[2] = pixel_center[2] - origin2;

            

            // ray_color func (we have origin and direction, which == ray)

            //for sphere in...
            // hit sphere
            float hit_sphere;
            float sphere_center[3];
            sphere_center[0] = 0.0;
            sphere_center[1] = 0.0;
            sphere_center[2] = -1.0;
            float sphere_radius = 0.5;
            float oc[3];
            oc[0] = sphere_center[0] - origin0;
            oc[1] = sphere_center[1] - origin1;
            oc[2] = sphere_center[2] - origin2;
            float a = direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]; //dot
            float b = -2.0 * (
                    direction[0]*oc[0] +
                    direction[1]*oc[1] +
                    direction[2]*oc[2]
                    );
            float c = (
                    oc[0]*oc[0] +
                    oc[1]*oc[1] +
                    oc[2]*oc[2] - sphere_radius*sphere_radius
                    );
            float discriminant = b*b - 4*a*c;
            if (discriminant < 0) {
                hit_sphere = -1.0;
            } else {
                hit_sphere = (-b - sqrt(discriminant)) / (2.0*a);
            }
            
            float t = hit_sphere;

            if (t > 0.0) {
                float N[3];
                N[0] = (origin0 + t*direction[0]) - 0.0;
                N[1] = (origin1 + t*direction[1]) - 0.0;
                N[2] = (origin2 + t*direction[2]) - -1.0;
                unitvector(N); // turns N into its unit vector

                color[0] = 0.5*(N[0] + 1.0);
                color[1] = 0.5*(N[1] + 1.0);
                color[2] = 0.5*(N[2] + 1.0);
            } else {
                // ray_color
                // unit_direction
                magnitude = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
                unit_direction[0] = direction[0]/magnitude;
                unit_direction[1] = direction[1]/magnitude;
                unit_direction[2] = direction[2]/magnitude;

                float a = 0.5*(unit_direction[1] + 1.0);

                color[0] = (1.0-a) * 1.0 + a * 0.5;
                color[1] = (1.0-a) * 1.0 + a * 0.7;
                color[2] = (1.0-a) * 1.0 + a * 1.0;
            }

            // // // // // // // write_color // // // // // // //
            screen_tex[idx] = color[0];
            screen_tex[idx + 1] = color[1];
            screen_tex[idx + 2] = color[2];

        }
    }
}

__managed__ unsigned char tex;

// instead of doing more samples I can just keep calling this over and over again!
// I just have to initialize R G B to zeroes
// then on each pass, I just add the color
// then at the end average them out..?
// has flaws at high sampling but whatever
extern "C" void render(
        int width, int height,
        const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
        const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
        const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
        const float origin0, const float origin1, const float origin2,
        int samples
        ) {
    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaError_t err;
    // size_t size = width * height * sizeof(float);

    // Launch the kernel
    renderKernel<<<numBlocks, threadsPerBlock>>>(
            &tex,
            width, height,
            pixel_delta_u0,
            pixel_delta_u1,
            pixel_delta_u2,
            pixel_delta_v0,
            pixel_delta_v1,
            pixel_delta_v2,
            pixel00_loc0,
            pixel00_loc1,
            pixel00_loc2,
            origin0,
            origin1,
            origin2,
            samples
            );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

}

 
void displayTexture() {
    glClear(GL_COLOR_BUFFER_BIT);
    printf("cleared\n");
    glBindTexture(GL_TEXTURE_2D, textureID);
    printf("texture bound\n");
    // draw a single quad covering the entire window
    // this is where the texture will be rendered
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd(); // end drawing the quad
    printf("quad drawn\n");
    glutSwapBuffers();
    printf("texture displayed\n");
}



void updateTexture() {
    size_t num_bytes;

    cudaGraphicsMapResources(1, &cudaResource, 0);
    printf("resources mapped\n");
    cudaGraphicsResourceGetMappedPointer((void**)&d_image, &num_bytes, cudaResource);
    printf("got mapped pointer\n");

    glBindTexture(target, textureID);
    printf("texture bound\n");
    glTexSubImage2D(target, 0, 0, 0, WIDTH, HEIGHT, GL_RGB32F, GL_UNSIGNED_BYTE, d_image);
    printf("subimage2d written\n");
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    printf("resources unmapped");
}



// initialize the *tex, the window, etc given parameters.
// 
extern "C" int initScene(
    int width, int height,
    const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
    const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
    const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
    const float origin0, const float origin1, const float origin2,
    int samples) {

    CHANNELS = 3;
    WIDTH = width;
    HEIGHT = height;
    target = GL_TEXTURE_2D;

    // Create a windowed mode window and its OpenGL context
    int argc = 0; // workaround for not passing argc and argv in via main()
    glutInit(&argc, NULL);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA OpenGL Interop");
    printf("Window created\n");

    // initialize glew
    glewInit();
    printf("glew initialized\n");

    //initOpenGL
    glEnable(target);
    glGenTextures(1, &textureID);
    glBindTexture(target, textureID);
    
    glTexImage2D(target, 0, GL_RGB32F, WIDTH, HEIGHT, 0, GL_RGB32F, GL_UNSIGNED_BYTE, NULL);
    
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    printf("OpenGL initiailized\n");

    //initCUDA
    cudaMalloc((void**)&d_image, width * height * CHANNELS * sizeof(unsigned char));
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderKernel<<<gridSize, blockSize>>>(
        d_image,
        width, height,
        pixel_delta_u0,
        pixel_delta_u1,
        pixel_delta_u2,
        pixel_delta_v0,
        pixel_delta_v1,
        pixel_delta_v2,
        pixel00_loc0,
        pixel00_loc1,
        pixel00_loc2,
        origin0,
        origin1,
        origin2,
        samples
    );
    cudaDeviceSynchronize();
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, target, cudaGraphicsRegisterFlagsNone);
    printf("cuda initialized\n");

    // other stuff    
    glutDisplayFunc(displayTexture);
    printf("texture display func set\n");
    glutIdleFunc(updateTexture);
    printf("texture update func set\n");
    glutMainLoop();

    cudaGraphicsUnregisterResource(cudaResource);
    return 0;
}

