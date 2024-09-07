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


#include <time.h>
#include <math.h>


// image
__managed__ int WIDTH;
__managed__ int HEIGHT;
__managed__ int CHANNELS;
__managed__ GLuint textureID;
__managed__ GLenum target;
__managed__ cudaGraphicsResource* cudaResource;
__managed__ unsigned char* d_image;
__managed__ unsigned int frame_count;
__managed__ time_t last_time;
__managed__ time_t now;

// game rendering
__managed__ int OBJ_COUNT; 



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


// ok.. this needs to be in THE EXACT SAME ORDER as zig
struct Sphere {
    float center[3];
    float radius;
};

struct Ray {
    float tmin;
    float tmax;
    float direction[3];
    float origin[3];
};

struct HitRecord {
    float p[3];
    float normal[3];
    float t;
    bool front_face;
};

__device__ float dot(float* v1, float* v2) {
    return (
            v1[0]*v2[0] +
            v1[1]*v2[1] +
            v1[2]*v2[2]
    );
}

// probably just unroll this lol it only shows up once...
// (OPTIMIZATION) CONVERT TO IF ELSE STATEMENT OR SOMETHING
// actually no dont
__device__ void set_face_normal(HitRecord* hit_record, Ray& r, float outward_normal[3]) {
    // cannot be float* because then you cannot assign to it.. I think? 
    hit_record->normal[0] = outward_normal[0]; 
    hit_record->normal[1] = outward_normal[1];
    hit_record->normal[2] = outward_normal[2];
    if (!(dot(r.direction, outward_normal) < 0)) {
       hit_record->normal[0] = hit_record->normal[0] * -1;
       hit_record->normal[1] = hit_record->normal[1] * -1;
       hit_record->normal[2] = hit_record->normal[2] * -1;
    }
}
 

// returns the point where the ray is at time=t
__device__ void at(Ray* r, float t, float position[3]) {
    position[0] = r->origin[0] + t*r->direction[0];
    position[1] = r->origin[1] + t*r->direction[1];
    position[2] = r->origin[2] + t*r->direction[2];
}


// the 'level' will be loaded into memory at init (one transfer from host to device)
// these are static objects
// in the future there will also be dynamic objects, a smaller array of objects that often update that is passed back and forth
// the goal is to minimize unnecessary transfers, since they are often one of the largest sources of inefficiency
__managed__ struct Sphere level_geometry[256]; // this may cause an error since most of these will be null...

// this is what gets run on the GPU
__global__ void renderKernel(
        unsigned char* screen_tex, int width, int height,
        const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2, // float* pixel_delta_u..
        const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
        const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
        const float origin0, const float origin1, const float origin2,
        int samples
        //dynamic_level_objects,
        ) {
        // int samples_per_pixel, int depth,
        // int hittables_flattened, int num_hittables) { //idk what to do about hittable type
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float pixel_center[3];
    float color[3]; 

    struct Ray ray;

    float unit_ray_direction[3];
    float magnitude;

    unsigned int idxr, idxg, idxb;
    
    float hit_sphere;
    //int level_geometry_length = sizeof(level_geometry)/sizeof(level_geometry[0]); 
    struct Sphere sphere; // should be a pointer probably (OPTIMIZATION)

    float outward_normal[3];
    float oc[3];
    float a, h, c, discriminant;
    float sqrtd, root;
    float N[3];

    ray.tmin = 0;
    ray.tmax = 10000.0;

    // execute threads from (0,0) to (height*ceil(sqrt(samples_per_pixel)), width*ceil(sqrt(samples_per_pixel)))
    // space has to fit each thread or whatever
    // int sqrt_samples_per_pixel = ceil(sqrt(samples_per_pixel)); // may cause issues affecting memory in outside executed threads...
    // one step at a time. first we just trace a single ray per pixel
    if (row < HEIGHT && col < WIDTH) {
        for (int i=0;i<samples;i++) {
            // create ray
            idxr = (col*WIDTH + row) * 4 + 0; // RGB - `texture` is [R, G, B, R, G, B, R, G, ...]
            idxg = (col*WIDTH + row) * 4 + 1;
            idxb =  (col*WIDTH + row) * 4 + 2;
            // // // // // // // ray color func here // // // // // // //
            // pixel's center in the viewport
            // uh, I think...
            pixel_center[0] = pixel00_loc0 + (row * pixel_delta_u0) + (col * pixel_delta_v0);
            pixel_center[1] = pixel00_loc1 + (row * pixel_delta_u1) + (col * pixel_delta_v1);
            pixel_center[2] = pixel00_loc2 + (row * pixel_delta_u2) + (col * pixel_delta_v2);

            ray.direction[0] = pixel_center[0] - origin0;
            ray.direction[1] = pixel_center[1] - origin1;
            ray.direction[2] = pixel_center[2] - origin2;

            ray.origin[0] = origin0;
            ray.origin[1] = origin1;
            ray.origin[2] = origin2;
            
            struct HitRecord hit_record;
            struct HitRecord temp_hit_record;
            bool hit_anything = false;
            float closest_so_far = ray.tmax;
            for (int s=0;s<OBJ_COUNT;s++){
                // ray_color func (we have origin and ray.direction, which == ray)
                // for sphere in spheres
                // hit sphere
                sphere = level_geometry[s]; // in the future, there will be more than spheres.
                oc[0] = sphere.center[0] - origin0; // oc = origin-to-center
                oc[1] = sphere.center[1] - origin1;
                oc[2] = sphere.center[2] - origin2;
                a = ray.direction[0]*ray.direction[0] + ray.direction[1]*ray.direction[1] + ray.direction[2]*ray.direction[2]; //dot
                h = (
                    ray.direction[0]*oc[0] +
                    ray.direction[1]*oc[1] +
                    ray.direction[2]*oc[2]
                );
                c = (
                    oc[0]*oc[0] +
                    oc[1]*oc[1] +
                    oc[2]*oc[2] - sphere.radius*sphere.radius
                );
                discriminant = h*h - a*c;
                if (discriminant < 0) {
                    continue;
                }

                sqrtd = sqrtf(discriminant);

                root = (h - sqrtd) / a;
                if (root <= ray.tmin || closest_so_far <= root) {
                    root = (h + sqrtd) / a;
                    if (root <= ray.tmin || closest_so_far <= root) {
                        continue;
                    }
                }
                
                temp_hit_record.t = root;
                at(
                        &ray,
                        temp_hit_record.t,
                        temp_hit_record.p
                        ); // overwrites p with the value
                temp_hit_record.normal[0] = (temp_hit_record.p[0] - sphere.center[0]) / sphere.radius;
                temp_hit_record.normal[1] = (temp_hit_record.p[1] - sphere.center[1]) / sphere.radius;
                temp_hit_record.normal[2] = (temp_hit_record.p[2] - sphere.center[2]) / sphere.radius;

                outward_normal[0] = (temp_hit_record.p[0] - sphere.center[0]) / sphere.radius;
                outward_normal[1] = (temp_hit_record.p[1] - sphere.center[1]) / sphere.radius;
                outward_normal[2] = (temp_hit_record.p[2] - sphere.center[2]) / sphere.radius;

                set_face_normal(&temp_hit_record, ray, outward_normal); 

                hit_sphere = true;
                
                closest_so_far = temp_hit_record.t;

            }
            
            hit_record = temp_hit_record;
                
            if (hit_sphere) {
                at(&ray, hit_record.t, N);
                // N[0] = (origin0 + t*ray.direction[0]) - 0.0;
                // N[1] = (origin1 + t*ray.direction[1]) - 0.0;
                // N[2] = (origin2 + t*ray.direction[2]) - -1.0;
                // printf("OTHER %f %f %f\n", origin0, t, ray.direction[0]);
                // printf("N %f %f %f\n", N[0], N[1], N[2]);

                unitvector(N); // turns N into its unit vector
                // printf("unit %f %f %f\n", N[0], N[1], N[2]);
                color[0] = 0.5*(N[0] + 1.0);
                color[1] = 0.5*(N[1] + 1.0);
                color[2] = 0.5*(N[2] + 1.0);
            } else { // if no sphere hit... draw sky
                // ray_color
                // unit_ray.direction
                magnitude = sqrt(
                        ray.direction[0]*ray.direction[0] +
                        ray.direction[1]*ray.direction[1] +
                        ray.direction[2]*ray.direction[2]
                );
                unit_ray_direction[0] = ray.direction[0]/magnitude;
                unit_ray_direction[1] = ray.direction[1]/magnitude;
                unit_ray_direction[2] = ray.direction[2]/magnitude;

                a = 0.5*(unit_ray_direction[1] + 1.0);

                color[0] = (1.0-a) * 1.0 + a * 0.5;
                color[1] = (1.0-a) * 1.0 + a * 0.7;
                color[2] = (1.0-a) * 1.0 + a * 1.0;
            }

            // // // // // // // write_color // // // // // // //
            screen_tex[idxr] = (unsigned char)(color[0]*255.0); // find some way to uh.. make this an unsigned char
            screen_tex[idxg] = (unsigned char)(color[1]*255.0);
            screen_tex[idxb] = (unsigned char)(color[2]*255.0);
            // printf("(%d, %d, %d)\n", screen_tex[idx], screen_tex[idx+1], screen_tex[idx+2]);
        }
    }
}

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
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

}

 
void glCheckError() {
    GLenum err = glGetError();
    if (err == GL_NO_ERROR) {
        printf("glCheckError(): NO ERROR\n");
    } else {
        printf("glCheckError(): %s\n", gluErrorString(err));
    }
}

void displayTexture() {
    glClear(GL_COLOR_BUFFER_BIT);
    // printf("displayTexture(): cleared\n");
    glBindTexture(GL_TEXTURE_2D, textureID);
    // printf("displayTexture(): texture bound\n");
    // draw a single quad covering the entire window
    // this is where the texture will be rendered
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glEnd(); // end drawing the quad
    // printf("displayTexture(): quad drawn\n");
    glutSwapBuffers();
    // printf("displayTexture(): texture displayed\n");
}


void updateTexture() {
    size_t num_bytes = WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char);
    // printf("updateTexture(): unsigned char size: %d\n", 8*sizeof(unsigned char));

    cudaGraphicsMapResources(1, &cudaResource, 0);
    // glCheckError();
    // printf("updateTexture(): resources mapped\n");
    // glCheckError();
    cudaArray_t cuda_Array; // inefficient. creates new array and then sends to device... every frame...
    cudaGraphicsSubResourceGetMappedArray(&cuda_Array, cudaResource, 0, 0);
    cudaMemcpyToArray(cuda_Array, 0, 0, d_image,num_bytes,cudaMemcpyDeviceToDevice);
    // printf("updateTexture(): got mapped pointer\n");

    cudaGraphicsUnmapResources(1, &cudaResource, 0);
    // printf("updateTexture(): resources unmapped\n");
    glutPostRedisplay();
}

// initialize the *tex, the window, etc given parameters.
// 
extern "C" int initScene(
    int width, int height,
    const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
    const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
    const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
    const float origin0, const float origin1, const float origin2,
    int samples,
    struct Sphere* level_geometry_host,
    int obj_count) {

    CHANNELS = 3;
    WIDTH = width;
    HEIGHT = height;
    target = GL_TEXTURE_2D;

    // set up FPS stuff
    frame_count = 0;
    time(&last_time);

    glCheckError();

    // Create a windowed mode window and its OpenGL context
    int argc = 0; // workaround for not passing argc and argv in via main()
    glutInit(&argc, NULL);
    glCheckError();
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // does this support GLUT_RGB32F?
    glCheckError();
    glutInitWindowSize(width, height);
    glCheckError();
    glutCreateWindow("CUDA OpenGL Interop");
    glCheckError();
    printf("main(): Window created\n");

    // initialize glew
    glewInit();
    printf("main(): glew initialized\n");
    glCheckError();

    //initOpenGL
    glEnable(target);
    glCheckError();
    glGenTextures(1, &textureID);
    glCheckError();
    glBindTexture(target, textureID);
    glCheckError();
    
    glTexImage2D(target, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL); //...
    glCheckError();
    
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glCheckError();
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glCheckError();
    printf("main(): OpenGL initiailized\n");
    glCheckError();


    // init level geometry/statics
    OBJ_COUNT = obj_count; // hardcoded for now. idc
    //for (int obj_idx=0;obj_idx<level_geometry_length;obj_idx++) {
    for (int obj_idx=0;obj_idx<OBJ_COUNT;obj_idx++) {
        // EACH OBJ IS A GUARANTEED SPHERE FOR NOW
        level_geometry[obj_idx] = level_geometry_host[obj_idx];
    }
    

    //initCUDA
    // MAKE SURE TO CHANGE IF TYPE OF TEXTURE ARRAY CHANGES - i.e. unsigned char -> float
    cudaMalloc((void**)&d_image, WIDTH * HEIGHT * CHANNELS * sizeof(unsigned char)); 
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
        //dynamic_level_objects,
    );
    cudaDeviceSynchronize();
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, target, cudaGraphicsRegisterFlagsNone);
    printf("main(): cuda initialized\n");

    // other stuff    
    // glutDisplayFunc(displayTexture);
    // glCheckError();
    // printf("main(): texture display func set\n");
    // glutIdleFunc(updateTexture);
    // glCheckError();
    // printf("main(): texture update func set\n");
    // glutMainLoop();

    // cudaGraphicsUnregisterResource(cudaResource);
    // cudaFree(d_image); // defer. should be separate function
    // checkBuffer();
    return 0;
}

extern "C" int render_scene(
    int width, int height,
    const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
    const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
    const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
    const float origin0, const float origin1, const float origin2,
    int samples) {
    // in the future... maybe keep objects on the device or something idk
    // and manipulate the memory from zig.. and run rendering on a loop

    // dim3 blockSize(16, 16);
    // dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);
    // dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    renderKernel<<<numBlocks, threadsPerBlock>>>(
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


    displayTexture();
    updateTexture();


    return 0;

}

