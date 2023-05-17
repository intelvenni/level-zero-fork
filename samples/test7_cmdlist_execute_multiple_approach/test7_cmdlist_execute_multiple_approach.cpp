/*
 *
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

 // TEST DESCRIPTION: Kernels appended in one command list executed by async queue, using every approach of appending kernel:
                      //zeCommandListAppendLaunchMultipleKernelsIndirect
                      //zeCommandListAppendLaunchCooperativeKernel
                      //zeCommandListAppendLaunchKernelIndirect
                      //zeCommandListAppendLaunchKernel

#include <stdlib.h>
#include "zello_init.h"

#include <memory>
#include <fstream>
#include <experimental/filesystem>

#define VALIDATECALL(myZeCall) \
    if (myZeCall != ZE_RESULT_SUCCESS){ \
        std::cout << "Error at "       \
            << #myZeCall << ": "       \
            << __FUNCTION__ << ": "    \
            << __LINE__ << std::endl;  \
        std::cout << "Exit with Error Code: " \
            << "0x" << std::hex \
            << myZeCall \
            << std::dec << std::endl; \
        std::terminate(); \
    }


// TODO: Ensure that all constructors are created similarly. Also ensure that the code follows some sort of styling.

int main(int argc, char* argv[])
{
    // Initialization
    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    // Driver initialization
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);

    ze_driver_handle_t driverHandle;
    zeDriverGet(&driverCount, &driverHandle);

    // Create the context
    ze_context_desc_t contextDescription = {};
    contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t context;
    zeContextCreate(driverHandle, &contextDescription, &context);

    // Device initialization
    uint32_t deviceCount = 0;
    zeDeviceGet(driverHandle, &deviceCount, nullptr);

    ze_device_handle_t device;
    zeDeviceGet(driverHandle, &deviceCount, &device);

    // Print device properties for debug purposes
    ze_device_properties_t deviceProperties = {};
    VALIDATECALL(zeDeviceGetProperties(device, &deviceProperties));
    std::cout << "\nDevice   : " << deviceProperties.name << "\n"
        << "Type     : " << ((deviceProperties.type == ZE_DEVICE_TYPE_GPU) ? "GPU" : "FPGA") << "\n"
        << "Vendor ID: " << std::hex << deviceProperties.vendorId << std::dec << "\n"
        << "maxMemAllocSize: " << deviceProperties.maxMemAllocSize << "\n";


    // Create a command queue
    uint32_t numQueueGroups = 0;
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
    if (numQueueGroups == 0) {
        std::cout << "No queue groups found\n";
        std::terminate();
    }
    else {
        std::cout << "#Queue Groups: " << numQueueGroups << std::endl;
    }
    std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data()));

    ze_command_queue_handle_t cmdQueue = nullptr;
    ze_command_queue_desc_t cmdQueueDesc = {};
    for (uint32_t i = 0; i < numQueueGroups; i++) {
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
        }
    }

    cmdQueueDesc.index = 0;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    VALIDATECALL(zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue));

    // Create a command list
    ze_command_list_handle_t cmdList = nullptr;
    ze_command_list_desc_t cmdListDesc = {};
    cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;
    VALIDATECALL(zeCommandListCreate(context, device, &cmdListDesc, &cmdList));


    // Create buffers
    const uint32_t items = 1024;
    constexpr size_t allocSize = items * items * sizeof(int);
    ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
    memAllocDesc.ordinal = 0;

    ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

    void* sharedA = nullptr;
    VALIDATECALL(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA));

    void* sharedB = nullptr;
    VALIDATECALL(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB));

    void* dstResultSum = nullptr;
    VALIDATECALL(zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum));

    // memory initialization
    int valA = 4;
    int valB = 2;
    memset(sharedA, valA, allocSize);
    memset(sharedB, valB, allocSize);

    // Module and kernel initialization
    ze_module_handle_t module;
    ze_kernel_handle_t kernels[2];
    ze_kernel_handle_t kernel1 = nullptr,
                       kernel2 = nullptr,
                       kernel3 = nullptr,
                       kernel4 = nullptr,
                       kernel5 = nullptr;

    kernels[0] = kernel1;
    kernels[1] = kernel2;

    std::ifstream file("matrixMultiply.spv", std::ios::binary);

    // zeCommandListAppendLaunchMultipleKernelsIndirect
    // #########################################################################################################
    // Open SPIR-V binary file
    if (file.is_open()) {
        file.seekg(0, file.end);
        auto length = file.tellg();
        file.seekg(0, file.beg);

        std::unique_ptr<char[]> spirvInput(new char[length]);
        file.read(spirvInput.get(), length);

        ze_module_desc_t moduleDesc = {};
        ze_module_build_log_handle_t buildLog;
        moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirvInput.get());
        moduleDesc.inputSize = length;
        moduleDesc.pBuildFlags = "";

        // Create module
        auto status = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
        if (status != ZE_RESULT_SUCCESS) {
            size_t szLog = 0;
            zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

            char* stringLog = (char*)malloc(szLog);
            zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
            std::cout << "Build log: " << stringLog << std::endl;
        }
        zeModuleBuildLogDestroy(buildLog);

        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;

        for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++)
        {
            // Create kernel
            ze_kernel_desc_t kernelDesc = {};
            kernelDesc.pKernelName = "incrementandsum";
            zeKernelCreate(module, &kernelDesc, &kernels[i]);

            zeKernelSuggestGroupSize(kernels[i], items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
            zeKernelSetGroupSize(kernels[i], groupSizeX, groupSizeY, groupSizeY);

            // Push arguments
            zeKernelSetArgumentValue(kernels[i], 0, sizeof(&sharedA), &sharedA);
            zeKernelSetArgumentValue(kernels[i], 1, sizeof(sharedB), &sharedB);
            zeKernelSetArgumentValue(kernels[i], 2, sizeof(dstResultSum), &dstResultSum);

        }

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        const uint32_t kernelArrSize = sizeof(kernels) / sizeof(kernels[0]);
        std::cout << "kernelArrSize: " << kernelArrSize << "\n";
        printf("We might get here...");
        zeCommandListAppendLaunchMultipleKernelsIndirect(cmdList, 2, kernels, &kernelArrSize, &launchArgs, nullptr, 0, nullptr);
        zeCommandListAppendLaunchKernelIndirect(cmdList, kernel4, &launchArgs, nullptr, 0, nullptr);
        printf("Do we get here?");

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }
    // #########################################################################################################

    // zeCommandListAppendLaunchCooperativeKernel
    // #########################################################################################################

    file.clear();
    file.open("matrixMultiply.spv", std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        auto length = file.tellg();
        file.seekg(0, file.beg);

        std::unique_ptr<char[]> spirvInput(new char[length]);
        file.read(spirvInput.get(), length);

        ze_module_desc_t moduleDesc = {};
        ze_module_build_log_handle_t buildLog;
        moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirvInput.get());
        moduleDesc.inputSize = length;
        moduleDesc.pBuildFlags = "";

        // Create module
        auto status = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
        if (status != ZE_RESULT_SUCCESS) {
            // TODO: Add an error print here and to other places
            size_t szLog = 0;
            zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

            char* stringLog = (char*)malloc(szLog);
            zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
            std::cout << "Build log: " << stringLog << std::endl;
        }
        zeModuleBuildLogDestroy(buildLog);


        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;

        // Create kernel
        ze_kernel_desc_t kernelDesc = {};
        kernelDesc.pKernelName = "incrementandsum";
        zeKernelCreate(module, &kernelDesc, &kernels[2]);

        zeKernelSuggestGroupSize(kernel3, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
        zeKernelSetGroupSize(kernel3, groupSizeX, groupSizeY, groupSizeY);

        // Push arguments
        zeKernelSetArgumentValue(kernel3, 0, sizeof(&sharedA), &sharedA);
        zeKernelSetArgumentValue(kernel3, 1, sizeof(sharedB), &sharedB);
        zeKernelSetArgumentValue(kernel3, 2, sizeof(dstResultSum), &dstResultSum);

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        zeCommandListAppendLaunchKernel(cmdList, kernel3, &launchArgs, nullptr, 0, nullptr);

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }
    // #########################################################################################################

    // zeCommandListAppendLaunchKernelIndirect
    // #########################################################################################################
    file.clear();
    file.open("matrixMultiply.spv", std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        auto length = file.tellg();
        file.seekg(0, file.beg);

        std::unique_ptr<char[]> spirvInput(new char[length]);
        file.read(spirvInput.get(), length);

        ze_module_desc_t moduleDesc = {};
        ze_module_build_log_handle_t buildLog;
        moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirvInput.get());
        moduleDesc.inputSize = length;
        moduleDesc.pBuildFlags = "";

        // Create module
        auto status = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
        if (status != ZE_RESULT_SUCCESS) {
            // TODO: Add an error print here and to other places
            size_t szLog = 0;
            zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

            char* stringLog = (char*)malloc(szLog);
            zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
            std::cout << "Build log: " << stringLog << std::endl;
        }
        zeModuleBuildLogDestroy(buildLog);


        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;

        // Create kernel
        ze_kernel_desc_t kernelDesc = {};
        kernelDesc.pKernelName = "incrementandsum";
        zeKernelCreate(module, &kernelDesc, &kernel4);

        zeKernelSuggestGroupSize(kernel4, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
        zeKernelSetGroupSize(kernel4, groupSizeX, groupSizeY, groupSizeY);

        // Push arguments
        zeKernelSetArgumentValue(kernel4, 0, sizeof(&sharedA), &sharedA);
        zeKernelSetArgumentValue(kernel4, 1, sizeof(sharedB), &sharedB);
        zeKernelSetArgumentValue(kernel4, 2, sizeof(dstResultSum), &dstResultSum);

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        zeCommandListAppendLaunchKernelIndirect(cmdList, kernel4, &launchArgs, nullptr, 0, nullptr);

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }
    // #########################################################################################################

    // zeCommandListAppendLaunchKernel
    // #########################################################################################################
    file.clear();
    file.open("matrixMultiply.spv", std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, file.end);
        auto length = file.tellg();
        file.seekg(0, file.beg);

        std::unique_ptr<char[]> spirvInput(new char[length]);
        file.read(spirvInput.get(), length);

        ze_module_desc_t moduleDesc = {};
        ze_module_build_log_handle_t buildLog;
        moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        moduleDesc.pInputModule = reinterpret_cast<const uint8_t*>(spirvInput.get());
        moduleDesc.inputSize = length;
        moduleDesc.pBuildFlags = "";

        // Create module
        auto status = zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
        if (status != ZE_RESULT_SUCCESS) {
            // TODO: Add an error print here and to other places
            size_t szLog = 0;
            zeModuleBuildLogGetString(buildLog, &szLog, nullptr);

            char* stringLog = (char*)malloc(szLog);
            zeModuleBuildLogGetString(buildLog, &szLog, stringLog);
            std::cout << "Build log: " << stringLog << std::endl;
        }
        zeModuleBuildLogDestroy(buildLog);


        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;

        // Create kernel
        ze_kernel_desc_t kernelDesc = {};
        kernelDesc.pKernelName = "incrementandsum";
        zeKernelCreate(module, &kernelDesc, &kernel5);

        zeKernelSuggestGroupSize(kernel5, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
        zeKernelSetGroupSize(kernel5, groupSizeX, groupSizeY, groupSizeY);

        // Push arguments
        zeKernelSetArgumentValue(kernel5, 0, sizeof(&sharedA), &sharedA);
        zeKernelSetArgumentValue(kernel5, 1, sizeof(sharedB), &sharedB);
        zeKernelSetArgumentValue(kernel5, 2, sizeof(dstResultSum), &dstResultSum);

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        zeCommandListAppendLaunchKernel(cmdList, kernel5, &launchArgs, nullptr, 0, nullptr);

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }
    // #########################################################################################################


    // Close list and submit for execution
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
    zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

    // Cleanup
    zeMemFree(context, dstResultSum);
    zeMemFree(context, sharedA);
    zeMemFree(context, sharedB);
    zeCommandListDestroy(cmdList);
    zeContextDestroy(context);
    zeCommandQueueDestroy(cmdQueue);

    std::cout << "\nTEST FINISHED\n";

    return 0;
}