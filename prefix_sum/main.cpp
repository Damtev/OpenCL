#include <iostream>
#include <CL/opencl.h>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <cmath>

#define N 1024 * 1024
#define PART_SIZE 64
#define FUNCTION_NAME "prefix_sum"
#define FILE_NAME "prefix_sum.cl"
#define MAX_DELTA log2(N)

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
    printf("Device: %s\r\n", value);
    free(value);

    // print hardware device version
    error_code = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, valueSize, value, NULL);
    printf(" %d. Hardware version: %s\r\n", 1, value);
    free(value);

    // print software driver version
    error_code = clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device driver version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, valueSize, value, NULL);
    printf(" %d. Software version: %s\r\n", 2, value);
    free(value);

    // print c version supported by compiler for device
    error_code = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device OpenCL version: %d", error_code);
        return;
    }
    value = (char *) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
    printf(" %d. OpenCL C version: %s\r\n", 3, value);
    free(value);

    // print parallel compute units
    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device max compute units: %d", error_code);
        return;
    }
    printf(" %d. Parallel compute units: %d\r\n", 4, maxComputeUnits);

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
    printf(" %d. Device type: %s\r\n", 5, deviceTypeName);
    free(deviceTypeName);

    error_code = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if (error_code != CL_SUCCESS) {
        printf("[ERROR] Getting device max compute units: %d", error_code);
        return;
    }
    printf(" %d. Parallel compute units: %d\r\n", 4, maxComputeUnits);
}

void print_kernel_time(cl_event &event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    printf("\tGlobal kernel time: %f ms\r\n", (end - start) * 1.0e-6f);
}

void check_correctness(const cl_float *in, const cl_float *prefix_sum, cl_uint n) {
    auto *naive_result = (cl_float *) malloc(n * sizeof(cl_float));

    for (int i = 0; i < n; ++i) {
        naive_result[i] = 0;
    }
    naive_result[0] = in[0];

    auto start_calculation = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < n; ++i) {
        naive_result[i] = naive_result[i - 1] + in[i];
    }

    auto end_calculation = std::chrono::high_resolution_clock::now();
    auto duration_calculation = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_calculation - start_calculation).count();


    for (size_t i = 0; i < n; ++i) {
        float delta = naive_result[i] - prefix_sum[i];
        float abs_delta = fabsf(delta);
        if (delta >= MAX_DELTA) {
            printf("\t[ERROR] Wrong computation: difference %f between naive %f and OpenCl %f at step %zu more than MAX_DELTA %f\r\n",
                   abs_delta, naive_result[i], prefix_sum[i], i, MAX_DELTA);
//            return;
        }
    }

    std::cout << "\r\n";
    std::cout << "\tNaive calculation time: " << duration_calculation << " ms" << "\r\n";
}

void opencl_prefix_sum() {
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
        int written = sprintf(buf, "-D PART_SIZE=%d", PART_SIZE);
        char options[written];
        memcpy(options, buf, written);

        auto err = clBuildProgram(program, 1, &device, options, NULL, NULL);
        if (err != 0) {
            std::cout << "\r\n\tError\r\n";
            size_t m_len;
            clGetProgramBuildInfo(program, (device), CL_PROGRAM_BUILD_LOG, 0, NULL,
                                  &m_len);
            char *log_buffer = (char *) calloc(m_len, sizeof(char));
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, m_len, log_buffer, &m_len);
            printf("\t%s", log_buffer);
        } else {
            std::cout << "\r\n\tSuccess\r\n";
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

            cl_uint n = N;

            auto mem_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n, NULL, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseMemObject(mem_in);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            auto mem_out = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * n, NULL, &error_code);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Creating buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                clReleaseMemObject(mem_in);
                clReleaseMemObject(mem_out);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            auto *in = (cl_float *) malloc(n * sizeof(cl_float));
            auto *prefix_sum = (cl_float *) malloc(n * sizeof(cl_float));

            for (int i1 = 0; i1 < n; ++i1) {
                in[i1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                in[i1] = 2.0f;
            }

            for (int i1 = 0; i1 < n; ++i1) {
                prefix_sum[i1] = 0;
            }

            error_code = clEnqueueWriteBuffer(queue, mem_in, CL_FALSE, 0, sizeof(cl_float) * n, in, 0,
                                              NULL,
                                              NULL);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Writing buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(in);
                free(prefix_sum);

                clReleaseMemObject(mem_in);
                clReleaseMemObject(mem_out);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_in);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_out);
            clSetKernelArg(kernel, 2, sizeof(cl_uint), &n);

            cl_uint work_dim = 1;
            auto *global_work_size = (size_t *) malloc(sizeof(size_t));
            global_work_size[0] = PART_SIZE;
            auto *local_work_size = (size_t *) malloc(sizeof(size_t));
            local_work_size[0] = PART_SIZE;

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

                free(in);
                free(prefix_sum);

                clReleaseMemObject(mem_in);
                clReleaseMemObject(mem_out);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }
            error_code = clEnqueueReadBuffer(queue, mem_out, true, 0, sizeof(cl_float) * n, prefix_sum, 0, NULL,
                                             NULL);
            if (error_code != CL_SUCCESS) {
                printf("[ERROR] Reading buffer: %d", error_code);
                free(platforms);
                free(gpus);
                free(cpus);
                free(buffer);

                free(in);
                free(prefix_sum);

                clReleaseMemObject(mem_in);
                clReleaseMemObject(mem_out);

                clReleaseKernel(kernel);
                clReleaseCommandQueue(queue);
                clReleaseProgram(program);
                clReleaseContext(context);
                return;
            }

            check_correctness(in, prefix_sum, n);
            print_kernel_time(event);

            free(in);
            free(prefix_sum);

            clReleaseMemObject(mem_in);
            clReleaseMemObject(mem_out);

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
    opencl_prefix_sum();

    return 0;
}