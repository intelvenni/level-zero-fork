/*
 *
 * Copyright (C) 2020-2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

// TEST DESCRIPTION: Reused command list executed in reused async command queue – queue synchronization needed.

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
        << "maxMemAllocSize: " << deviceProperties.maxMemAllocSize << "\n\n";

    
    // Create a command queue
    uint32_t numQueueGroups = 0;
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr));
    if (numQueueGroups == 0) {
        std::cout << "No queue groups found\n";
        std::terminate();
    } else {
        std::cout << "#Queue Groups: " << numQueueGroups << std::endl;
    }
    std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
    VALIDATECALL(zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data()));

    ze_command_queue_handle_t cmdQueue;
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
    ze_command_list_handle_t cmdList;
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
    ze_module_handle_t module = nullptr;
    ze_kernel_handle_t kernel = nullptr;

    std::ifstream file("matrixMultiply.spv", std::ios::binary);

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

        // Create kernel
        ze_kernel_desc_t kernelDesc = {};
        kernelDesc.pKernelName = "incrementandsum";
        zeKernelCreate(module, &kernelDesc, &kernel);

        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;
        zeKernelSuggestGroupSize(kernel, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
        zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeY);

        // Push arguments
        zeKernelSetArgumentValue(kernel, 0, sizeof(&sharedA), &sharedA);
        zeKernelSetArgumentValue(kernel, 1, sizeof(sharedB), &sharedB);
        zeKernelSetArgumentValue(kernel, 2, sizeof(dstResultSum), &dstResultSum);

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        zeCommandListAppendLaunchKernel(cmdList, kernel, &launchArgs, nullptr, 0, nullptr);

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }

    // Close list and submit for execution
    zeCommandListClose(cmdList);
    zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
    zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

    // Reset (recycle) command list for new commands
    zeCommandListReset(cmdList);

    std::cout << "\nAppending and execution done once.\n" << std::endl;

    // Do the whole kernel execution process again but with reused command lists and command queues
    // The new values for the kernel arguments will be taken from the values that the kernel returns for dtsResultSum
    // Open SPIR-V binary file
    std::ifstream file2("matrixMultiply.spv", std::ios::binary);

    if (file2.is_open()) {
        file2.seekg(0, file2.end);
        auto length = file2.tellg();
        file2.seekg(0, file2.beg);

        std::unique_ptr<char[]> spirvInput(new char[length]);
        file2.read(spirvInput.get(), length);

        uint32_t groupSizeX = 32u;
        uint32_t groupSizeY = 32u;
        uint32_t groupSizeZ = 1u;
        zeKernelSuggestGroupSize(kernel, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
        zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeY);

        // Push arguments
        // Push output of dstResultSum to both A and B arguments. The resulting sum should be:
        // (A + 1) + (B + 2) OR (9 + 1) + (9 + 2) = 21
        memset(sharedA, *((int*)dstResultSum), allocSize);
        memset(sharedB, *((int*)dstResultSum), allocSize);
        zeKernelSetArgumentValue(kernel, 0, sizeof(&sharedA), &sharedA);
        zeKernelSetArgumentValue(kernel, 1, sizeof(sharedB), &sharedB);
        zeKernelSetArgumentValue(kernel, 2, sizeof(dstResultSum), &dstResultSum);

        // Kernel thread-dispatch
        ze_group_count_t launchArgs;
        launchArgs.groupCountX = items / groupSizeX;
        launchArgs.groupCountY = items / groupSizeY;
        launchArgs.groupCountZ = 1;

        // Immediately submit a kernel to the device and launch
        zeCommandListAppendLaunchKernel(cmdList, kernel, &launchArgs, nullptr, 0, nullptr);

        file.close();
    }
    else {
        std::cout << "SPIR-V binary file not found\n";
        std::terminate();
    }

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