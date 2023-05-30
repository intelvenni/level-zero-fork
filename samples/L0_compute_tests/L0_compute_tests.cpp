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
	ze_command_list_handle_t cmdList2 = nullptr;
	ze_command_queue_desc_t cmdQueueDesc = {};
	std::vector<ze_module_handle_t> modules;
	std::vector<ze_kernel_handle_t> kernels;

	L0ComputeTest(std::string testName, std::string testDescription, ze_device_handle_t& device, ze_context_handle_t& context, bool syncWithEvent, int kernelAmount, int moduleAmount, bool secondCmdListNeeded, bool immediateCmdListNeeded, bool commandListReused, bool syncCommandQueue, _ze_command_queue_mode_t cmdQueueMode, kernelLaunchScenario kernelScenario) {

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

		std::cout << "Creating command lists" << "\n";
		if (immediateCmdListNeeded) {
			createImmediateCmdList(context, device, numQueueGroups, queueProperties, cmdList);
		}
		else {
			createCmdQueue(context, device, cmdQueue, cmdQueueMode, numQueueGroups, queueProperties, cmdQueueDesc);
			createCommandList(context, device, numQueueGroups, queueProperties, cmdList, cmdQueueDesc);
			if (secondCmdListNeeded)
			{
				createCommandList(context, device, numQueueGroups, queueProperties, cmdList2, cmdQueueDesc);
			}
		}

		// Create buffers
		std::cout << "Creating buffers" << "\n";
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

		// Create modules
		std::cout << "Creating modules" << "\n";
		for (size_t i = 0; i < moduleAmount; i++)
		{
			ze_module_handle_t module = nullptr;
			createModule(context, device, module);
			modules.push_back(module);
		}

		// Create kernels
		std::cout << "Creating kernels" << "\n";
		for (size_t i = 0; i < kernelAmount; i++)
		{
			ze_kernel_handle_t kernel;
			ze_kernel_desc_t kernelDesc = {};
			kernelDesc.pKernelName = "incrementandsum";
			zeKernelCreate(modules[i], &kernelDesc, &kernel);
			kernels.push_back(kernel);
		}

		// If test uses event to sync: Create the event
		ze_event_handle_t event = nullptr;
		if (syncWithEvent) {
			std::cout << "Creating event" << "\n";
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

		// Launch kernels
		// If test re-uses command list: Run the cycle twice
		std::cout << "Launching kernels" << "\n";
		int commandListLoop = 1;
		if (commandListReused)
			int commandListLoop = 2;
		for (size_t i = 0; i < commandListLoop; i++)
		{
			for (size_t i = 0; i < kernelAmount; i++)
			{
				if (secondCmdListNeeded && i > 0)
					appendAndLaunchKernels(event, cmdList2, kernels[i], kernelScenario, items, sharedA, sharedB, dstResultSum);
				else
					appendAndLaunchKernels(event, cmdList, kernels[i], kernelScenario, items, sharedA, sharedB, dstResultSum);
			}

			// If immediate command list is not used: Close list and submit for execution
			if (!immediateCmdListNeeded)
			{
				std::cout << "Closing command lists" << "\n";
				zeCommandListClose(cmdList);
				if (secondCmdListNeeded)
					zeCommandListClose(cmdList2);
				zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, nullptr);
				if (secondCmdListNeeded)
					zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList2, nullptr);

				// If command queue needs to be synchronized
				if (syncCommandQueue)
					zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());

				// If test re-uses command list: Reset (recycle) command list for new commands
				if (commandListReused) {
					std::cout << "Recycling command list" << "\n";
					zeCommandQueueSynchronize(cmdQueue, std::numeric_limits<uint64_t>::max());
					zeCommandListReset(cmdList);
				}
			}
		}

		// Validation
		uint32_t* srcA = static_cast<uint32_t*>(sharedA);
		uint32_t* srcB = static_cast<uint32_t*>(sharedB);
		uint32_t* dstResult = static_cast<uint32_t*>(dstResultSum);

		std::cout << "\n\nsrcA value: " << srcA[0] << std::endl;
		std::cout << "srcB value: " << srcB[0] << std::endl;
		std::cout << "dstResult value: " << dstResult[0] << std::endl;
		if (kernelScenario != ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT)
			validateSum(srcA, srcB, dstResult);


		// Cleanup
		zeMemFree(context, dstResultSum);
		zeMemFree(context, sharedA);
		zeMemFree(context, sharedB);
		zeCommandListDestroy(cmdList);
		if (cmdQueue != nullptr)
			zeCommandQueueDestroy(cmdQueue);

		std::cout << "Test status: PASS" << "\n\n";
	}
};

void validateSum(uint32_t* srcA, uint32_t* srcB, uint32_t* result) {

	int sum = 0;
	sum = srcA[0] + srcB[0] + 1;
	std::cout << "Validation: " << (*result == sum ? "PASSED" : "FAILED") << "\n";
}

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

// Create a command list
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

// Create kernel and module
void createModule(ze_context_handle_t& context, ze_device_handle_t& device, ze_module_handle_t& module) {
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
		file.close();
	}
	else {
		std::cout << "SPIR-V binary file not found\n";
		std::cout << "\nTest status: FAIL\n";
		std::terminate();
	}

	file.close();

}

// Append and launch kernels
void appendAndLaunchKernels(ze_event_handle_t& event, ze_command_list_handle_t& cmdList, ze_kernel_handle_t& kernel, kernelLaunchScenario kernelScenario, uint32_t items, void*& buffA, void*& buffB, void*& buffResult) {
	uint32_t groupSizeX = 32u;
	uint32_t groupSizeY = 32u;
	uint32_t groupSizeZ = 1u;
	zeKernelSuggestGroupSize(kernel, items, items, 1U, &groupSizeX, &groupSizeY, &groupSizeZ);
	zeKernelSetGroupSize(kernel, groupSizeX, groupSizeY, groupSizeY);

	// Push arguments
	zeKernelSetArgumentValue(kernel, 0, sizeof(&buffA), &buffA);
	zeKernelSetArgumentValue(kernel, 1, sizeof(&buffB), &buffB);
	zeKernelSetArgumentValue(kernel, 2, sizeof(&buffResult), &buffResult);

	// Kernel thread-dispatch
	ze_group_count_t launchArgs;
	launchArgs.groupCountX = items / groupSizeX;
	launchArgs.groupCountY = items / groupSizeY;
	launchArgs.groupCountZ = 1;

	// Immediately submit a kernel to the device and launch
	// Event may be nullptr if sync is not needed
	switch (kernelScenario)
	{
	case ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL:
		zeCommandListAppendLaunchKernel(cmdList, kernel, &launchArgs, event, 0, nullptr);
		break;
	case ZE_COMMAND_LIST_APPEND_LAUNCH_MULTIPLE_KERNELS_INDIRECT:
		//zeCommandListAppendLaunchMultipleKernelsIndirect(cmdList, 2, kernels, &kernelArrSize, &launchArgs, nullptr, 0, nullptr); // TODO: This launching scenario does not work. Why? (Program freezes and eventually dies.)
		break;
	case ZE_COMMAND_LIST_APPEND_LAUNCH_COOPERATIVE_KERNEL:
		zeCommandListAppendLaunchCooperativeKernel(cmdList, kernel, &launchArgs, event, 0, nullptr);
		break;
	case ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT:
		zeCommandListAppendLaunchKernelIndirect(cmdList, kernel, &launchArgs, event, 0, nullptr);
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	ze_device_handle_t device = nullptr;
	ze_context_handle_t context = nullptr;
	initializeDeviceAndContext(context, device);

	L0ComputeTest testAppendAsyncImmeadiateCmdListSyncEvent("testAppendAsyncImmeadiateCmdListSyncEvent", "Kernel appended to async immediate command list - synchronization with signal event.", device, context, true, 1, 1, false, true, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendSyncImmediateCmdListNoSync("testAppendSyncImmediateCmdListNoSync", "Kernel appended to synchronous immediate command list - synchronization not needed.", device, context, false, 1, 1, false, true, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendAsyncCmdListExecAsyncCmdQueueSync("testAppendAsyncCmdListExecAsyncCmdQueueSync", "Kernel appended to async immediate command list - queue synchronization needed.", device, context, false, 1, 1, false, false, false, true, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendCmdListSyncCmdQueue("testAppendCmdListSyncCmdQueue", "Kernel appended to command list executed with sync command queue - synchronization not needed.", device, context, false, 1, 1, false, false, false, false, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendReusedCmdListAndCmdQueue("testAppendReusedCmdListAndCmdQueue", "Reused command list executed in reused async command queue - queue synchronization needed.", device, context, false, 1, 1, false, false, true, true, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendTwoKernelsDiffCmdListSameCmdQueue("testAppendTwoKernelsDiffCmdListSameCmdQueue", "Two kernels appended to different command lists, both executed with same async command queue - queue synchronization needed.", device, context, false, 2, 2, true, false, false, true, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendOneCmdListMultipleApproaches1("testAppendOneCmdListMultipleApproaches1", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - zeCommandListAppendLaunchCooperativeKernel.", device, context, false, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_COOPERATIVE_KERNEL);
	L0ComputeTest testAppendOneCmdListMultipleApproaches2("testAppendOneCmdListMultipleApproaches2", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - zeCommandListAppendLaunchKernelIndirect.", device, context, false, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT);
	L0ComputeTest testAppendOneCmdListMultipleApproaches3("testAppendOneCmdListMultipleApproaches3", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - zeCommandListAppendLaunchKernel.", device, context, false, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);
	L0ComputeTest testAppendOneCmdListMultipleApproachesSyncWithEvent1("testAppendOneCmdListMultipleApproachesSyncWithEvent1", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - sync - zeCommandListAppendLaunchCooperativeKernel.", device, context, true, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_COOPERATIVE_KERNEL);
	L0ComputeTest testAppendOneCmdListMultipleApproachesSyncWithEvent2("testAppendOneCmdListMultipleApproachesSyncWithEvent2", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - sync - zeCommandListAppendLaunchKernelIndirect.", device, context, true, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT);
	L0ComputeTest testAppendOneCmdListMultipleApproachesSyncWithEvent3("testAppendOneCmdListMultipleApproachesSyncWithEvent3", "Kernels appended in one command list executed by async queue, using every approach of appending kernel - sync - zeCommandListAppendLaunchKernel.", device, context, true, 2, 2, false, false, false, false, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL);

	// Final cleanup
	zeContextDestroy(context);

	return 0;
}