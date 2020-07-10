#include <iostream>
#include <CL/opencl.h>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <cmath>

#define TILE_SIZE 32
#define PER_THREAD 8
#define N 2048 // first matrix [N x M]
#define M 512 // second matrix [M x L]
#define L 1024 // result matrix [N x L]
#define FUNCTION_NAME "matrix_mul"
#define FILE_NAME "matrix_mult_per_thread.cl"
#define MAX_DELTA 0.05

std::string get_device_type(cl_device_type deviceType) {
    switch (deviceType) {
        case CL_DEVICE_TYPE_CPU:
            return "CPU";
        case CL_DEVICE_TYPE_GPU:
            return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR:
            return "ACCELERATOR";
        case CL_DEVICE_TYPE_DEFAULT:
            return "DEFAULT";
        default:
            return "Unknown device type";
    }
}

void print_device_info(cl_device_id device) {
    char *value;
    size_t valueSize;
    cl_uint maxComputeUnits;
    cl_device_type deviceType;
    char *deviceTypeName;


    cl_int error_code = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device name: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);
    free(value);

    // print hardware device version
    error_code = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
    printf(" %d. Hardware version: %s\n", 1, value);
    free(value);

    // print software driver version
    error_code = clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device driver version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
    printf(" %d. Software version: %s\n", 2, value);
    free(value);

    // print c version supported by compiler for device
    error_code = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device OpenCL version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf(" %d. OpenCL C version: %s\n", 3, value);
    free(value);

    // print parallel compute units
    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device max compute units: %d", error_code);
        return;
    }
    printf(" %d. Parallel compute units: %d\n", 4, maxComputeUnits);

    // print device type
    error_code = clGetDeviceInfo(device, CL_DEVICE_TYPE,
                                 sizeof(deviceType), &deviceType, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device type: %d", error_code);
        return;
    }
    std::string temp = get_device_type(deviceType);
    deviceTypeName = (char *) malloc(temp.size());
    strcpy(deviceTypeName, temp.c_str());
    printf(" %d. Device type: %s\n", 5, deviceTypeName);
    free(deviceTypeName);

    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device max compute units: %d", error_code);
        return;
    }
    printf(" %d. Parallel compute units: %d\n", 4, maxComputeUnits);
}

void print_kernel_specifications(cl_event &event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    cl_ulong time = end - start;
    printf("\tGlobal kernel time: %f ms\r\n", time * 1.0e-6f);
    cl_ulong operations = 2 * N;
    operations *= M;
    operations *= L;
    cl_ulong gflops = operations / time;
    printf("\tGflops: %lu\r\n", gflops);
}

void check_correctness(const cl_float *first_matrix, const cl_float *second_matrix, cl_float *result_matrix) {
    auto *naive_result = (cl_float *) malloc(N * L * sizeof(cl_float));

    for (int i1 = 0; i1 < N; ++i1) {
        for (int j1 = 0; j1 < L; ++j1) {
            naive_result[i1 * L + j1] = 0;
        }
    }

    auto start_calculation = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            for (int j = 0; j < L; ++j) {
                naive_result[i * L + j] += first_matrix[i * M + k] * second_matrix[k * L + j];
            }
        }
    }

    auto end_calculation = std::chrono::high_resolution_clock::now();
    auto duration_calculation = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_calculation - start_calculation).count();


    for (size_t i = 0; i < N * L; ++i) {
            float delta = naive_result[i] - result_matrix[i];
            float abs_delta = fabsf(delta);

            if (delta >= MAX_DELTA) {
                printf("[ERROR] Wrong computation: difference %f between naive %f and OpenCl %f at step %zu more than MAX_DELTA %f\r\n",
                       abs_delta, naive_result[i], result_matrix[i], i, MAX_DELTA);
            }
    }

    std::cout << "\n";
    std::cout << "\tOpenMP calculation time: " << duration_calculation << " ms" << std::endl;
}



void opencl_mult_matrices() {
    srand(time(nullptr));
    int i;
    cl_uint platformCount;
    cl_platform_id *platforms;
    cl_uint gpuCount;
    cl_uint cpuCount;
    cl_device_id *gpus;
    cl_device_id *cpus;
    cl_int error_code;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * platformCount);
    error_code = clGetPlatformIDs(platformCount, platforms, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting platforms: %d", error_code);
        free(platforms);
        return;
    }

    cl_device_id discrete_gpu = nullptr;
    cl_device_id gpu = nullptr;
    cl_device_id cpu = nullptr;
    cl_device_id device;
    if (platformCount > 0) {
        for (i = 0; i < platformCount; i++) {
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &gpuCount);
            gpus = (cl_device_id *) malloc(sizeof(cl_device_id) * gpuCount);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, NULL);

            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &cpuCount);
            cpus = (cl_device_id *) malloc(sizeof(cl_device_id) * cpuCount);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, gpuCount, cpus, NULL);

            if (gpuCount > 0 && discrete_gpu == nullptr) {
                gpu = gpus[0];
                for (int j = 0; j < gpuCount; ++j) {
                    cl_bool has_unified_memory;
                    clGetDeviceInfo(gpus[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &has_unified_memory, NULL);
                    if (has_unified_memory == CL_FALSE) {
                        discrete_gpu = gpus[j];
                        break;
                    }
                }
            }
            if (cpu == nullptr && cpuCount > 0) {
                cpu = cpus[0];
            }
        }
        device = (discrete_gpu) ? discrete_gpu : ((gpu) ? gpu : cpu);
        print_device_info(device);
        auto context =
                clCreateContext(NULL, 1, &device, NULL, NULL, &error_code);
        if (error_code != CL_SUCCESS) {
            printf("[ERROR] Creating context: %d", error_code);
            free(platforms);
            free(gpus);
            free(cpus);

            clReleaseContext(context);
            return;
        }

        auto queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error_code);
        if (error_code != CL_SUCCESS) {
            printf("[ERROR] Creating command queue: %d", error_code);
            free(platforms);
            free(gpus);
            free(cpus);

            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            return;
        }

        FILE *fileptr;
        char *buffer;
        long filelen;

        fileptr = fopen(FILE_NAME, "rb");  // Open the file in binary mode
        fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
        filelen = ftell(fileptr);             // Get the current byte offset in the file
        rewind(fileptr);                      // Jump back to the beginning of the file

        buffer = (char *) malloc(filelen * sizeof(char)); // Enough memory for the file
        fread(buffer, filelen, 1, fileptr); // Read in the entire file
        fclose(fileptr); // Close the file

        const char *cbuffer = buffer;
        const size_t clen = filelen;

        auto program = clCreateProgramWithSource(context, 1, (&cbuffer), (&clen), &error_code);
        if (error_code != CL_SUCCESS) {
            printf("[ERROR] Creating program: %d", error_code);
            free(platforms);
            free(gpus);
            free(cpus);
            free(buffer);

            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            return;
        }
        char buf[80];
        int written = sprintf(buf, "-D TILE_SIZE=%d -D PER_THREAD=%d", TILE_SIZE, PER_THREAD);
        char options[written];
        memcpy(options, buf, written);

        auto err = clBuildProgram(program, 1, &device, options, NULL, NULL);
        if (err != 0) {
            std::cout << "\n\tError\n";
            size_t m_len;
            clGetProgramBuildInfo(program, (device), CL_PROGRAM_BUILD_LOG, 0, NULL,
                                  &m_len);
            char *log_buffer = (char *) calloc(m_len, sizeof(char));
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, m_len, log_buffer, &m_len);
            printf("\t%s", log_buffer);
        } else {
            std::cout << "\n\tSuccess\n";
            auto kernel = clCreateKernel(program, FUNCTION_NAME, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating kernel: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseKernel(kernel);
                clReleaseProgram(program);
                clReleaseCommandQueue(queue);
                clReleaseContext(context);
                return;
            }

            int n = N;
            int m = M;
            int l = L;
            auto mem_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n * m, NULL, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseMemObject(mem_a);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            auto mem_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * m * l, NULL, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            auto mem_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n * l, NULL, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);
                clReleaseMemObject(mem_c);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            auto *first_matrix = (cl_float *) malloc(n * m * sizeof(cl_float));
            auto *second_matrix = (cl_float *) malloc(m * l * sizeof(cl_float));
            auto *result_matrix = (cl_float *) malloc(n * l * sizeof(cl_float));

            for (int i1 = 0; i1 < n * m; ++i1) {
                first_matrix[i1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }

            for (int i1 = 0; i1 < m * l; ++i1) {
                second_matrix[i1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            }

            for (int i1 = 0; i1 < n * l; ++i1) {
                result_matrix[i1] = 0;
            }

            error_code = clEnqueueWriteBuffer(queue, mem_a, CL_FALSE, 0, sizeof(cl_float) * n * m, first_matrix, 0,
                                              NULL,
                                              NULL);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Writing buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(first_matrix);
                free(second_matrix);
                free(result_matrix);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);
                clReleaseMemObject(mem_c);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            error_code = clEnqueueWriteBuffer(queue, mem_b, CL_FALSE, 0, sizeof(cl_float) * m * l, second_matrix, 0,
                                              NULL,
                                              NULL);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Writing buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(first_matrix);
                free(second_matrix);
                free(result_matrix);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);
                clReleaseMemObject(mem_c);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_a);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_b);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_c);
            clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
            clSetKernelArg(kernel, 4, sizeof(cl_uint), &m);
            clSetKernelArg(kernel, 5, sizeof(cl_uint), &l);

            cl_uint work_dim = 2;
            auto *global_work_size = (size_t *) malloc(2 * sizeof(size_t));
            global_work_size[0] = l; // y
            global_work_size[1] = n / PER_THREAD; // x
            auto *local_work_size = (size_t *) malloc(2 * sizeof(size_t));
            local_work_size[0] = TILE_SIZE;
            local_work_size[1] = TILE_SIZE / PER_THREAD;

            cl_event event;
            error_code = clEnqueueNDRangeKernel(queue, kernel, work_dim, 0, global_work_size,
                                                local_work_size, NULL,
                                                NULL,
                                                &event);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Enquing kernel: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(first_matrix);
                free(second_matrix);
                free(result_matrix);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);
                clReleaseMemObject(mem_c);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            error_code = clEnqueueReadBuffer(queue, mem_c, true, 0, sizeof(cl_float) * n * l, result_matrix, 0, NULL,
                                             NULL);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Reading buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(first_matrix);
                free(second_matrix);
                free(result_matrix);

                clReleaseMemObject(mem_a);
                clReleaseMemObject(mem_b);
                clReleaseMemObject(mem_c);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            check_correctness(first_matrix, second_matrix, result_matrix);
            print_kernel_specifications(event);

            free(first_matrix);
            free(second_matrix);
            free(result_matrix);

            clReleaseMemObject(mem_a);
            clReleaseMemObject(mem_b);
            clReleaseMemObject(mem_c);

            clReleaseKernel(kernel);
            clReleaseCommandQueue(queue);
            clReleaseProgram(program);
            clReleaseContext(context);
        }

        free(buffer);

        free(cpus);
        free(gpus);
        free(platforms);
    } else {
        std::cout << "No OpenCL platform found";
    }
}

int main() {
    opencl_mult_matrices();

    return 0;
}
