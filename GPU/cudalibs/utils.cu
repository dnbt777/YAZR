#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fillMatKernel(float *M, int width, int height, float c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // execute threads from (0, 0) to (height, width)
    if (row < height && col < width) {
        M[row * width + col] = c; // notice a pointer to a float is passed in, but we are indexing it
        // this is intentional and is how we access other elements of the matrix
    }
}


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

__managed__ float d_R[1000 * 1000], d_G[1000*1000], d_B[1000*1000];

__global__ void shootRaysKernel(
        int width, int height,
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
            // // // // // // // ray color func here // // // // // // //
            // pixel's center in the viewport
            // uh, I think...
            pixel_center[0] = pixel00_loc0 + (row * pixel_delta_u0) + (col * pixel_delta_v0),
            pixel_center[1] = pixel00_loc1 + (row * pixel_delta_u1) + (col * pixel_delta_v1),
            pixel_center[2] = pixel00_loc2 + (row * pixel_delta_u2) + (col * pixel_delta_v2),

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
            d_R[row*width + col] = color[0]*255.0;
            d_G[row*width + col] = color[1]*255.0;
            d_B[row*width + col] = color[2]*255.0;

        }
    }
}




extern "C" void initImage(float *R, float *G, float *B, int width, int height) {
    // float *d_R, *d_G, *d_B;
    size_t size = width * height * sizeof(float);

    // Allocate device memory and check for errors
    cudaError_t errR, errG, errB, err;
    errR = cudaMalloc((void**)&d_R, size);
    errG = cudaMalloc((void**)&d_G, size);
    errB = cudaMalloc((void**)&d_B, size);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for img: %s\n", cudaGetErrorString(errR));
        cudaFree(d_R);
        cudaFree(d_G);
        cudaFree(d_B);
        return;
    }

    // Copy matrices from host to device
    errR = cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
    errG = cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
    errB = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error copying matrix to device: %s\n", cudaGetErrorString(errR));
        cudaFree(d_R);
        cudaFree(d_G);
        cudaFree(d_B);
        return;
    }
}


// instead of doing more samples I can just keep calling this over and over again!
// I just have to initialize R G B to zeroes
// then on each pass, I just add the color
// then at the end average them out..?
// has flaws at high sampling but whatever
extern "C" void shootRays(
        float *R, float *G, float *B, int width, int height,
        const float pixel_delta_u0,const float pixel_delta_u1,const float pixel_delta_u2,
        const float pixel_delta_v0,const float pixel_delta_v1,const float pixel_delta_v2,
        const float pixel00_loc0,const float pixel00_loc1,const float pixel00_loc2,
        const float origin0, const float origin1, const float origin2,
        int samples
        ) {
    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaError_t errR, errG, errB, err;
    size_t size = width * height * sizeof(float);

    // Launch the kernel
    shootRaysKernel<<<numBlocks, threadsPerBlock>>>(
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

    // Copy result from device to host
    // errR = cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);
    // errG = cudaMemcpy(G, d_G, size, cudaMemcpyDeviceToHost);
    // errB = cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error copying C to host: %s\n", cudaGetErrorString(errR));
    }

    // Free device memory
    //cudaFree(d_R);
    //cudaFree(d_G);
    //cudaFree(d_B);
}







__global__ void colorImgKernel(float *R, float *G, float *B, int width, int height) {

    // R G and B channels are passed in as separate parameters (obviously ig)

    // get row and col of thread. move to block, then move to thread in the block => get abs thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // row * width + col => each row increases index by width. each col increases that by 1
    if (row < height && col < width) {
        R[row * width + col] = 255.0 * float(row) / float(width - 1);
        G[row * width + col] = 255.0 * float(col) / float(height - 1);
        B[row * width + col] = 0.0;
    }
}



__global__ void matmulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

extern "C" void matmul(float *A, float *B, float *C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate device memory and check for errors
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for A: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }
    
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy matrices from host to device
    err = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying A to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    err = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying B to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    matmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result from device to host
    err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying C to host: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" void colorImg(float *R, float *G, float *B, int width, int height) {
    float *d_R, *d_G, *d_B;
    size_t size = width * height * sizeof(float);

    // Allocate device memory and check for errors
    cudaError_t errR, errG, errB, err;
    errR = cudaMalloc((void**)&d_R, size);
    errG = cudaMalloc((void**)&d_G, size);
    errB = cudaMalloc((void**)&d_B, size);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for img: %s\n", cudaGetErrorString(errR));
        cudaFree(d_R);
        cudaFree(d_G);
        cudaFree(d_B);
        return;
    }

    // Copy matrices from host to device
    errR = cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
    errG = cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
    errB = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error copying matrix to device: %s\n", cudaGetErrorString(errR));
        cudaFree(d_R);
        cudaFree(d_G);
        cudaFree(d_B);
        return;
    }


    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    colorImgKernel<<<numBlocks, threadsPerBlock>>>(d_R, d_G, d_B, width, height);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result from device to host
    errR = cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);
    errG = cudaMemcpy(G, d_G, size, cudaMemcpyDeviceToHost);
    errB = cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
    if (errR != cudaSuccess || errG != cudaSuccess || errB != cudaSuccess) {
        fprintf(stderr, "Error copying C to host: %s\n", cudaGetErrorString(errR));
    }

    // Free device memory
    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
}

extern "C" void fillMat(float *M, int width, int height, float c) {
    float *d_M;
    size_t size = width * height * sizeof(float);

    // Allocate device memory and check for errors
    cudaError_t err;
    err = cudaMalloc((void**)&d_M, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for matrix: %s\n", cudaGetErrorString(err));
        return;
    }

    // Copy matrices from host to device
    err = cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying matrix to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_M);
        return;
    }


    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    fillMatKernel<<<numBlocks, threadsPerBlock>>>(d_M, width, height, c);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Copy result from device to host
    err = cudaMemcpy(M, d_M, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying C to host: %s\n", cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_M);
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
