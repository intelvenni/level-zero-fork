/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#pragma once

enum kernelLaunchScenario {
	ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL,
	ZE_COMMAND_LIST_APPEND_LAUNCH_MULTIPLE_KERNELS_INDIRECT,
	ZE_COMMAND_LIST_APPEND_LAUNCH_COOPERATIVE_KERNEL,
	ZE_COMMAND_LIST_APPEND_LAUNCH_KERNEL_INDIRECT
};

void validateSum(uint32_t* srcA, uint32_t* srcB, uint32_t* result);
void initializeDeviceAndContext(ze_context_handle_t& context, ze_device_handle_t& device);
void createCmdQueue(ze_context_handle_t& context, ze_device_handle_t& device, ze_command_queue_handle_t& cmdQueue, ze_command_queue_mode_t cmdQueueMode, uint32_t& numQueueGroups, std::vector<ze_command_queue_group_properties_t>& queueProperties, ze_command_queue_desc_t& cmdQueueDesc);
void createCommandList(ze_context_handle_t& context, ze_device_handle_t& device, uint32_t numQueueGroups, std::vector<ze_command_queue_group_properties_t> queueProperties, ze_command_list_handle_t& cmdList, ze_command_queue_desc_t& cmdQueueDesc);
void createImmediateCmdList(ze_context_handle_t& context, ze_device_handle_t& device, uint32_t numQueueGroups, std::vector<ze_command_queue_group_properties_t> queueProperties, ze_command_list_handle_t& cmdList);
void createModule(ze_context_handle_t& context, ze_device_handle_t& device, ze_module_handle_t& module);
void appendAndLaunchKernels(ze_event_handle_t& event, ze_command_list_handle_t& cmdList, ze_kernel_handle_t& kernel, kernelLaunchScenario kernelScenario, uint32_t items, void*& buffA, void*& buffB, void*& buffResult);