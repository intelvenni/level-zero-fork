/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <stdlib.h>
#include "zello_init.h"
#include "L0_compute_tests.h"

#include <memory>
#include <fstream>


class L0ComputeTest {
public:

	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	ze_kernel_handle_t kernels[2];
	ze_module_handle_t modules[2];

	L0ComputeTest(std::string testName, std::string testDescription, ze_device_handle_t& device, ze_context_handle_t& context, bool syncWithEvent, int kernelAmount, int moduleAmount, bool immediateCmdListNeeded, kernelLaunchScenario kernelScenario) {

		std::cout << "Running test: " << testName << "\n";
		std::cout << "Description: " << testDescription << "\n";

		uint32_t numQueueGroups = 0;
		zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
		if (numQueueGroups == 0) {
			std::cout << "No queue groups found\n";
			std::terminate();
		}

		std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
		zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data());

		if (immediateCmdListNeeded) {
			createImmediateCmdList(context, device, numQueueGroups, queueProperties, cmdList);
		}
		else {
			createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
			createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);
		}

		// Create buffers
		const uint32_t items = 1024;
		constexpr size_t allocSize = items * items * sizeof(int);
		ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
		memAllocDesc.ordinal = 0;

		ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

		void* sharedA = nullptr;
		zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

		void* sharedB = nullptr;
		zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

		void* dstResultSum = nullptr;
		zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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

			ze_event_handle_t event = nullptr;
			if (syncWithEvent) {
				// Create event and event pool
				ze_event_pool_desc_t eventPoolDesc = {
					ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
					nullptr,
					ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // All events in pool are visible to Host
					1
				};
				ze_event_pool_handle_t eventPool;
				zeEventPoolCreate(context, &eventPoolDesc, 0, nullptr, &eventPool);

				ze_event_desc_t eventDesc = {
					ZE_STRUCTURE_TYPE_EVENT_DESC,
					nullptr,
					0,
					0,
					ZE_EVENT_SCOPE_FLAG_HOST
				};
				
				zeEventCreate(eventPool, &eventDesc, &event);
			}

			// Immediately submit a kernel to the device and launch
			// Event may be nullptr if sync is not needed
			switch (kernelScenario)
			{
			case ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL:
				zeCommandListAppendLaunchKernel(cmdList, kernel, &launchArgs, event, 0, nullptr);
				break;
			case ZE_COMMAND_LIST_APPEND_LAUNCH_MULTIPLE_KERNELS_INDIRECT:
				//zeCommandListAppendLaunchMultipleKernelsIndirect(cmdList, 2, kernels, &kernelArrSize, &launchArgs, nullptr, 0, nullptr);
				break;
			case ZE_COMMAND_LIST_APPEND_LAUNCH_COOPERATIVE_KERNEL:
				zeCommandListAppendLaunchCooperativeKernel(cmdList, kernel, &launchArgs, nullptr, 0, nullptr); // TODO: Muuta kernel muuttuja oikeaksi
				break;
			case ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT:
				zeCommandListAppendLaunchKernelIndirect(cmdList, kernel, &launchArgs, nullptr, 0, nullptr); // TODO: Muuta kernel muuttuja oikeaksi
				break;
			default:
				break;
			}
			
			file.close();
		}
		else {
			std::cout << "SPIR-V binary file not found\n";
			std::cout << "\nTest status: FAIL\n";
			std::terminate();
		}

		std::cout << "Test status: PASS" << "\n\n";
	}
};

 // Init, create device and context
void initializeDeviceAndContext(ze_context_handle_t& context, ze_device_handle_t& device) {

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

	zeContextCreate(driverHandle, &contextDescription, &context);

	// Device initialization
	uint32_t deviceCount = 0;
	zeDeviceGet(driverHandle, &deviceCount, nullptr);

	zeDeviceGet(driverHandle, &deviceCount, &device);

}

// Create a command queue
void createCmdQueue(ze_context_handle_t& context, ze_device_handle_t& device, ze_command_queue_handle_t& cmdQueue, ze_command_queue_mode_t cmdQueueMode, uint32_t& numQueueGroups, std::vector<ze_command_queue_group_properties_t>& queueProperties, ze_command_queue_desc_t& cmdQueueDesc) {

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data());


	for (uint32_t i = 0; i < numQueueGroups; i++) {
		if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
			cmdQueueDesc.ordinal = i;
		}
	}

	cmdQueueDesc.index = 0;
	cmdQueueDesc.mode = cmdQueueMode;
	zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue);
}

void createCommandList(ze_context_handle_t& context, ze_device_handle_t& device, uint32_t numQueueGroups, std::vector<ze_command_queue_group_properties_t> queueProperties, ze_command_list_handle_t& cmdList, ze_command_queue_desc_t& cmdQueueDesc) {

	ze_command_list_desc_t cmdListDesc = {};
	cmdListDesc.commandQueueGroupOrdinal = cmdQueueDesc.ordinal;
	zeCommandListCreate(context, device, &cmdListDesc, &cmdList);
}

// Create an immediate command list (An immediate command list is both a command list and an implicit command queue.) 
void createImmediateCmdList(ze_context_handle_t& context, ze_device_handle_t& device, uint32_t numQueueGroups, std::vector<ze_command_queue_group_properties_t> queueProperties, ze_command_list_handle_t& cmdList) {

	ze_command_queue_desc_t commandQueueDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
		nullptr,
		0,
		0, // index 
		0, // flags 
		ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
		ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	};

	for (uint32_t i = 0; i < numQueueGroups; i++) {
		if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
			commandQueueDesc.ordinal = i;
		}
	}

	zeCommandListCreateImmediate(context, device, &commandQueueDesc, &cmdList);
}


void testAppendAsyncImmeadiateCmdListSyncEvent(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCIPRTION: Kernel appended to async immediate command list – synchronization with signal event

	uint32_t numQueueGroups = 0;
	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data());

	ze_command_list_handle_t cmdList = nullptr;
	createImmediateCmdList(context, device, numQueueGroups, queueProperties, cmdList);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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

		// Create event and event pool
		ze_event_pool_desc_t eventPoolDesc = {
			ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
			nullptr,
			ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // All events in pool are visible to Host
			1
		};
		ze_event_pool_handle_t eventPool;
		zeEventPoolCreate(context, &eventPoolDesc, 0, nullptr, &eventPool);

		ze_event_desc_t eventDesc = {
			ZE_STRUCTURE_TYPE_EVENT_DESC,
			nullptr,
			0,
			0,
			ZE_EVENT_SCOPE_FLAG_HOST
		};
		ze_event_handle_t event;
		zeEventCreate(eventPool, &eventDesc, &event);

		// Immediately submit a kernel to the device and launch
		zeCommandListAppendLaunchKernel(cmdList, kernel, &launchArgs, event, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 1 FAILED\n";
		std::terminate();
	}

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdList);

	std::cout << "\nTEST 1 FINISHED\n";
}

void testAppendSyncImmeadiateCmdListNoSync(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCIPRTION: Kernel appended to synchronous immediate command list – synchronization not needed

	uint32_t numQueueGroups = 0;
	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);
	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties.data());

	ze_command_list_handle_t cmdList = nullptr;
	createImmediateCmdList(context, device, numQueueGroups, queueProperties, cmdList);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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
		std::cout << "\nTEST 2 FAILED\n";
		std::terminate();
	}

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdList);

	std::cout << "\nTEST 2 FINISHED\n";
}

void testAppendAsyncCmdListExecAsyncCmdQueueSync(ze_context_handle_t& context, ze_device_handle_t& device) {

	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
	createImmediateCmdList(context, device, numQueueGroups, queueProperties, cmdList);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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
		std::cout << "\nTEST 3 FAILED\n";
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
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 3 FINISHED\n";
}

void testAppendCmdListSyncCmdQueue(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCRIPTION: Kernel appended to command list executed with sync command queue – synchronization not needed.

	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
	createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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
		std::cout << "\nTEST 4 FAILED\n";
		std::terminate();
	}

	// Close list and submit for execution
	zeCommandListClose(cmdList);
	zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdList);
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 4 FINISHED\n";
}

void testAppendReusedCmdListAndCmdQueue(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCRIPTION: Reused command list executed in reused async command queue – queue synchronization needed.

	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
	createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);


	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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
		std::cout << "\nTEST 5 FAILED\n";
		std::terminate();
	}

	// Close list and submit for execution
	zeCommandListClose(cmdList);
	zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
	zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

	// Reset (recycle) command list for new commands
	zeCommandListReset(cmdList);

	// Do the whole kernel execution process again but with reused command lists and command queues
	// The new values for the kernel arguments will be taken from the values that the kernel returns for dstResultSum
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
		// Push output of dstResultSum to both A and B arguments
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
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 5 FINISHED\n";
}

void testAppendTwoKernelsDiffCmdListSameCmdQueue(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCRIPTION: Two kernels appended to different command lists, both executed with same async command queue – queue synchronization needed.

	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);

	// Create two command lists
	ze_command_list_handle_t cmdLists[2] = { nullptr, nullptr };

	createCommandList(context, device, numQueueGroups, queueProperties, cmdLists[0], cmdQueueDesc);
	createCommandList(context, device, numQueueGroups, queueProperties, cmdLists[1], cmdQueueDesc);


	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

	// memory initialization
	int valA = 4;
	int valB = 2;
	memset(sharedA, valA, allocSize);
	memset(sharedB, valB, allocSize);

	// Module and kernel initialization
	ze_module_handle_t module1, module2 = nullptr;
	ze_kernel_handle_t kernel1, kernel2 = nullptr;
	std::vector<ze_module_handle_t> modules;
	std::vector<ze_kernel_handle_t> kernels;

	modules.push_back(module1);
	modules.push_back(module2);
	kernels.push_back(kernel1);
	kernels.push_back(kernel2);

	for (size_t i = 0; i < modules.size(); i++)
	{
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
			auto status = zeModuleCreate(context, device, &moduleDesc, &modules[i], &buildLog);
			if (status != ZE_RESULT_SUCCESS) {
				// TODO: Add an error print here and to other places
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
			zeKernelCreate(modules[i], &kernelDesc, &kernels[i]);

			uint32_t groupSizeX = 32u;
			uint32_t groupSizeY = 32u;
			uint32_t groupSizeZ = 1u;
			zeKernelSuggestGroupSize(kernels[i], items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
			zeKernelSetGroupSize(kernels[i], groupSizeX, groupSizeY, groupSizeY);

			// Push arguments
			zeKernelSetArgumentValue(kernels[i], 0, sizeof(&sharedA), &sharedA);
			zeKernelSetArgumentValue(kernels[i], 1, sizeof(sharedB), &sharedB);
			zeKernelSetArgumentValue(kernels[i], 2, sizeof(dstResultSum), &dstResultSum);

			// Kernel thread-dispatch
			ze_group_count_t launchArgs;
			launchArgs.groupCountX = items / groupSizeX;
			launchArgs.groupCountY = items / groupSizeY;
			launchArgs.groupCountZ = 1;

			// Immediately submit a kernel to the device and launch
			zeCommandListAppendLaunchKernel(cmdLists[i], kernels[i], &launchArgs, nullptr, 0, nullptr);
			zeCommandListClose(cmdLists[i]);

			file.close();
		}
		else {
			std::cout << "SPIR-V binary file not found\n";
			std::cout << "\nTEST 6 FAILED\n";
			std::terminate();
		}
	}

	// Close list and submit for execution
	zeCommandQueueExecuteCommandLists(cmdQueue, 2, cmdLists, nullptr);
	zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdLists[0]);
	zeCommandListDestroy(cmdLists[1]);
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 6 FINISHED\n";

}

void testAppendOneCmdListMultipleApproaches(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCRIPTION: Kernels appended in one command list executed by async queue, using every approach of appending kernel:

	//zeCommandListAppendLaunchMultipleKernelsIndirect
	//zeCommandListAppendLaunchCooperativeKernel
	//zeCommandListAppendLaunchKernelIndirect
	//zeCommandListAppendLaunchKernel


	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
	createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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

	/*
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
		zeCommandListAppendLaunchMultipleKernelsIndirect(cmdList, 2, kernels, &kernelArrSize, &launchArgs, nullptr, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 7.1 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	*/

	// zeCommandListAppendLaunchCooperativeKernel
	// #########################################################################################################

	//file.clear();
	//file.open("matrixMultiply.spv", std::ios::binary);

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
		zeKernelCreate(module, &kernelDesc, &kernel3);

		uint32_t groupSizeX = 32u;
		uint32_t groupSizeY = 32u;
		uint32_t groupSizeZ = 1u;
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
		zeCommandListAppendLaunchCooperativeKernel(cmdList, kernel3, &launchArgs, nullptr, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 7.2 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 7.2 DONE\n";

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
		std::cout << "\nTEST 7.3 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 7.3 DONE\n";

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
		std::cout << "\nTEST 7.4 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 7.4 DONE\n";


	// Close list and submit for execution
	zeCommandListClose(cmdList);
	zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
	zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdList);
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 7 FINISHED\n";
}

void testAppendOneCmdListMultipleApproachesSyncEvent(ze_context_handle_t& context, ze_device_handle_t& device) {
	// TEST DESCRIPTION: Kernels appended in one command list executed by async queue, using every approach of appending kernel - sync with event:

	//zeCommandListAppendLaunchMultipleKernelsIndirect
	//zeCommandListAppendLaunchCooperativeKernel
	//zeCommandListAppendLaunchKernelIndirect
	//zeCommandListAppendLaunchKernel


	ze_command_queue_handle_t cmdQueue = nullptr;
	ze_command_list_handle_t cmdList = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	uint32_t numQueueGroups = 0;

	zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, nullptr);
	if (numQueueGroups == 0) {
		std::cout << "No queue groups found\n";
		std::terminate();
	}

	std::vector<ze_command_queue_group_properties_t> queueProperties(numQueueGroups);

	createCmdQueue(context, device, cmdQueue, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, numQueueGroups, queueProperties, cmdQueueDesc);
	createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);

	// Create buffers
	const uint32_t items = 1024;
	constexpr size_t allocSize = items * items * sizeof(int);
	ze_device_mem_alloc_desc_t memAllocDesc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
	memAllocDesc.ordinal = 0;

	ze_host_mem_alloc_desc_t hostDesc = { ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC };

	void* sharedA = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedA);

	void* sharedB = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &sharedB);

	void* dstResultSum = nullptr;
	zeMemAllocShared(context, &memAllocDesc, &hostDesc, allocSize, 1, device, &dstResultSum);

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

	// Create event and event pool
	ze_event_pool_desc_t eventPoolDesc = {
		ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
		nullptr,
		ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // All events in pool are visible to Host
		1
	};
	ze_event_pool_handle_t eventPool;
	zeEventPoolCreate(context, &eventPoolDesc, 0, nullptr, &eventPool);

	ze_event_desc_t eventDesc = {
		ZE_STRUCTURE_TYPE_EVENT_DESC,
		nullptr,
		0,
		0,
		ZE_EVENT_SCOPE_FLAG_HOST
	};
	ze_event_handle_t event;
	zeEventCreate(eventPool, &eventDesc, &event);

	std::ifstream file("matrixMultiply.spv", std::ios::binary);

	/*
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
		zeCommandListAppendLaunchMultipleKernelsIndirect(cmdList, 2, kernels, &kernelArrSize, &launchArgs, event, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 8.1 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	*/

	// zeCommandListAppendLaunchCooperativeKernel
	// #########################################################################################################

	//file.clear();
	//file.open("matrixMultiply.spv", std::ios::binary);

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
		zeKernelCreate(module, &kernelDesc, &kernel3);

		uint32_t groupSizeX = 32u;
		uint32_t groupSizeY = 32u;
		uint32_t groupSizeZ = 1u;
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
		zeCommandListAppendLaunchCooperativeKernel(cmdList, kernel3, &launchArgs, event, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 8.2 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 8.2 DONE\n";

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
		zeCommandListAppendLaunchKernelIndirect(cmdList, kernel4, &launchArgs, event, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 8.3 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 8.3 DONE\n";

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
		zeCommandListAppendLaunchKernel(cmdList, kernel5, &launchArgs, event, 0, nullptr);

		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTEST 8.4 FAILED\n";
		std::terminate();
	}
	// #########################################################################################################
	std::cout << "\nTEST 8.4 DONE\n";


	// Close list and submit for execution
	zeCommandListClose(cmdList);
	zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
	zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

	// Cleanup
	zeMemFree(context, dstResultSum);
	zeMemFree(context, sharedA);
	zeMemFree(context, sharedB);
	zeCommandListDestroy(cmdList);
	zeCommandQueueDestroy(cmdQueue);

	std::cout << "\nTEST 8 FINISHED\n";
}

int main(int argc, char* argv[])
{
	ze_device_handle_t device = nullptr;
	ze_context_handle_t context = nullptr;

	initializeDeviceAndContext(context, device);

	testAppendAsyncImmeadiateCmdListSyncEvent(context, device);
	testAppendSyncImmeadiateCmdListNoSync(context, device);
	testAppendAsyncCmdListExecAsyncCmdQueueSync(context, device);
	testAppendCmdListSyncCmdQueue(context, device);
	testAppendReusedCmdListAndCmdQueue(context, device);
	testAppendTwoKernelsDiffCmdListSameCmdQueue(context, device);
	testAppendOneCmdListMultipleApproaches(context, device);
	testAppendOneCmdListMultipleApproachesSyncEvent(context, device);

	// Final cleanup
	zeContextDestroy(context);

	return 0;
}