# coding:utf-8
import json
import random
from collections import Counter
from scipy import stats
import pandas as pd
import argparse
import time
import numpy as np
import scipy

parser = argparse.ArgumentParser(description='Measurement')
parser.add_argument('--GDP_addr', type=str, default='./input_data/GDP_USA.json')
parser.add_argument('--geographic_addr', type=str, default='./input_data/geographic_USA.json')
parser.add_argument('--bandwidth_distribution_addr', type=str, default='./data/bandwidth_distribution.json')
parser.add_argument('--cpu_distribution_addr', type=str, default='./data/cpu_distribution.json')
parser.add_argument('--memory_distribution_addr', type=str, default='./data/memory_distribution.json')
parser.add_argument('--disk_distribution_addr', type=str, default='./data/disk_distribution.json')
parser.add_argument('--devices_out_addr', type=str, default='./out/Edge_devices_'+str(time.time())+'.csv')
parser.add_argument('--resource_bias_precision', type=int, default=3.5)
parser.add_argument('--geo_bias_precision', type=int, default=0.5)
args = parser.parse_args()

def load_json(addr):
    with open(addr, 'r') as load_f:
        load_data = json.load(load_f)
        return load_data

def cal_spearman(x1, y1):
    X1 = np.array(x1)
    Y1 = np.array(y1)
    return scipy.stats.spearmanr(X1, Y1).correlation

def cal_bias_resource(disk_region_list, memory_region_list, cpu_region_list, bandwidth_region_list):
    total_bias = 0
    total_bias += abs(cal_spearman(disk_region_list, cpu_region_list) - 0.15)
    total_bias += abs(cal_spearman(cpu_region_list, memory_region_list) - 0.42)
    total_bias += abs(cal_spearman(cpu_region_list, bandwidth_region_list) - 0.42)
    total_bias += abs(cal_spearman(disk_region_list, memory_region_list) - 0.29)
    total_bias += abs(cal_spearman(disk_region_list, bandwidth_region_list) - 0.36)
    total_bias += abs(cal_spearman(memory_region_list, bandwidth_region_list) - 0.44)
    return total_bias

def cal_bias_geo(GDP_data, device_resource_list):
    GDP_list = []
    disk_list = []
    bandwidth_list = []
    memory_list = []
    cpu_list = []
    for region in device_resource_list:
        GDP_list.append(GDP_data[region])
        device_list_region = device_resource_list[region]
        disk_list.append(sum([one[0] for one in device_list_region]))
        memory_list.append(sum([one[1] for one in device_list_region]))
        cpu_list.append(sum([one[2] for one in device_list_region]))
        bandwidth_list.append(sum([one[3] for one in device_list_region]))
    total_bias = 0
    total_bias += abs(cal_spearman(GDP_list, cpu_list) - 0.84)
    total_bias += abs(cal_spearman(cpu_list, disk_list) - 0.89)
    total_bias += abs(cal_spearman(cpu_list, memory_list) - 0.84)
    total_bias += abs(cal_spearman(cpu_list, bandwidth_list) - 0.80)
    return total_bias

# step 1: The goal of this step is to take the total GDP or population of a region as input to output the total amount
#         of each type of resource for the whole region.
def ESMG_step1(GDP_data):
    print('Step 1 - start')
    total_cpu = {}
    total_memory = {}
    total_disk = {}
    total_bandwidth = {}
    for i in GDP_data:
        # It is calculated based on fitting equations
        GDP_billion = GDP_data[i] * (10 ** -3)
        total_cpu[i] = GDP_billion * 1.206 * (10 ** 1) - 1.341 * (10 ** 3)
        total_memory[i] = GDP_billion * 1.906 * (10 ** 10) - 1.947 * (10 ** 12)
        total_disk[i] = GDP_billion * 3.316 * (10 ** 12) - 5.657 * (10 ** 14)
        total_bandwidth[i] = GDP_billion * 8.086 * (10 ** 8) - 7.387 * (10 ** 10)
    print('End')
    return total_cpu, total_memory, total_disk, total_bandwidth


# step 2: The goal of this step is to use the total amount of each type of resource in the region as input to output
#         the number of edge servers in this region.
def ESMG_step2(total_cpu, total_memory, total_disk, total_bandwidth):
    print('Step 2 - start')
    device_num_cpu = {}
    device_num_memory = {}
    device_num_disk = {}
    device_num_bandwidth = {}
    device_num = {}

    # The total resources are divided by the average resources of a device to calculate the number of devices
    for i in total_cpu:
        device_num_cpu[i] = total_cpu[i] / 11.51007
        device_num_memory[i] = total_memory[i] / (17.51910 * 1024 * 1024 * 1024)
        device_num_disk[i] = total_disk[i] / (2456.53572 * 1024 * 1024 * 1024)
        device_num_bandwidth[i] = total_bandwidth[i] / (782.75923 * 1024 * 1024)
        device_num[i] = int(
            (device_num_cpu[i] + device_num_memory[i] + device_num_disk[i] + device_num_bandwidth[i]) / 4)
    print('End')
    return device_num


# step 3: The goal of this step is to take the total number of each resource and the number of edge servers in the
#         region as input to output the distribution of each resource in this region.
def ESMG_step3(total_cpu, total_memory, total_disk, total_bandwidth, device_num):
    print('Step 3 - start')
    disk_distribution = load_json(args.disk_distribution_addr)
    cpu_distribution = load_json(args.cpu_distribution_addr)
    memory_distribution = load_json(args.memory_distribution_addr)
    bandwidth_distribution = load_json(args.bandwidth_distribution_addr)

    disk_distribution_each_region = {}
    cpu_distribution_each_region = {}
    memory_distribution_each_region = {}
    bandwidth_distribution_each_region = {}

    for region in device_num:
        disk_distribution_tmp = []
        memory_distribution_tmp = []
        cpu_distribution_tmp = []
        bandwidth_distribution_tmp = []

        # Calculate the ratio of the resource amount of the target region and the resource amount of the dataset
        disk_resource_ratio = total_disk[region] / float(sum(disk_distribution))
        memory_resource_ratio = total_memory[region] / float(sum(memory_distribution))
        cpu_resource_ratio = total_cpu[region] / float(sum(cpu_distribution))
        bandwidth_resource_ratio = total_bandwidth[region] / float(sum(bandwidth_distribution))

        # Update the data according to the ratio of the previous step
        disk_distribution_adv = [one_disk * disk_resource_ratio for one_disk in disk_distribution]
        memory_distribution_adv = [one_memory * memory_resource_ratio for one_memory in memory_distribution]
        cpu_distribution_adv = [one_cpu * cpu_resource_ratio for one_cpu in cpu_distribution]
        bandwidth_distribution_adv = [one_bandwidth * bandwidth_resource_ratio for one_bandwidth in
                                      bandwidth_distribution]

        disk_counter = dict(Counter(disk_distribution_adv))
        memory_counter = dict(Counter(memory_distribution_adv))
        cpu_counter = dict(Counter(cpu_distribution_adv))
        bandwidth_counter = dict(Counter(bandwidth_distribution_adv))

        # Split resources to each device - disk
        for type in disk_counter:
            disk_proportion = device_num[region] * (disk_counter[type] / float(len(disk_distribution_adv)))
            for num_same in range(int(disk_proportion)):
                disk_distribution_tmp.append(type * disk_counter[type] / disk_proportion)
            if disk_proportion - int(disk_proportion) != 0:
                disk_distribution_tmp.append(
                    type * disk_counter[type] / disk_proportion * (disk_proportion - int(disk_proportion)))
        # Split resources to each device - memory
        for type in memory_counter:
            memory_proportion = device_num[region] * (memory_counter[type] / float(len(memory_distribution_adv)))
            for num_same in range(int(memory_proportion)):
                memory_distribution_tmp.append(type * memory_counter[type] / memory_proportion)
            if memory_proportion - int(memory_proportion) != 0:
                memory_distribution_tmp.append(
                    type * memory_counter[type] / memory_proportion * (memory_proportion - int(memory_proportion)))
        # Split resources to each device - cpu
        for type in cpu_counter:
            cpu_proportion = device_num[region] * (cpu_counter[type] / float(len(cpu_distribution_adv)))
            for num_same in range(int(cpu_proportion)):
                cpu_distribution_tmp.append(type * cpu_counter[type] / cpu_proportion)
            if cpu_proportion - int(cpu_proportion) != 0 and (type * cpu_counter[type] / cpu_proportion * (cpu_proportion - int(cpu_proportion))) != 0:
                cpu_distribution_tmp.append(
                    type * cpu_counter[type] / cpu_proportion * (cpu_proportion - int(cpu_proportion)))
        # Split resources to each device - bandwidth
        for type in bandwidth_counter:
            bandwidth_proportion = device_num[region] * (
                    bandwidth_counter[type] / float(len(bandwidth_distribution_adv)))
            for num_same in range(int(bandwidth_proportion)):
                bandwidth_distribution_tmp.append(type * bandwidth_counter[type] / bandwidth_proportion)
            if bandwidth_proportion - int(bandwidth_proportion) != 0:
                bandwidth_distribution_tmp.append(type * bandwidth_counter[type] / bandwidth_proportion * (
                        bandwidth_proportion - int(bandwidth_proportion)))

        # Make the number of elements in different resource lists consistent.
        vaild_num = min(len(disk_distribution_tmp), len(memory_distribution_tmp), len(cpu_distribution_tmp),
                        len(bandwidth_distribution_tmp))
        disk_distribution_tmp.sort(reverse=True)
        disk_distribution_tmp = disk_distribution_tmp[:vaild_num]
        memory_distribution_tmp.sort(reverse=True)
        memory_distribution_tmp = memory_distribution_tmp[:vaild_num]
        cpu_distribution_tmp.sort(reverse=True)
        cpu_distribution_tmp = cpu_distribution_tmp[:vaild_num]
        bandwidth_distribution_tmp.sort(reverse=True)
        bandwidth_distribution_tmp = bandwidth_distribution_tmp[:vaild_num]

        disk_distribution_each_region[region] = disk_distribution_tmp
        memory_distribution_each_region[region] = memory_distribution_tmp
        cpu_distribution_each_region[region] = cpu_distribution_tmp
        bandwidth_distribution_each_region[region] = bandwidth_distribution_tmp
    print('End')
    return disk_distribution_each_region, memory_distribution_each_region, cpu_distribution_each_region, bandwidth_distribution_each_region


# step 4: The goal of this step is to take the distribution of each resource in the region as input to output the
#         resource configuration for each edge server in this region.
def ESMG_step4(disk_distribution_each_region, memory_distribution_each_region, cpu_distribution_each_region,
               bandwidth_distribution_each_region):
    print('Step 4 - start')
    device_resource_list = {}
    for region in disk_distribution_each_region:
        disk_region_list = disk_distribution_each_region[region]
        memory_region_list = memory_distribution_each_region[region]
        cpu_region_list = cpu_distribution_each_region[region]
        bandwidth_region_list = bandwidth_distribution_each_region[region]

        # Calculate the correlation coefficient bias with the dataset
        cur_total_bias = cal_bias_resource(disk_region_list, memory_region_list, cpu_region_list, bandwidth_region_list)
        pre_total_bias = cur_total_bias
        flag = 0

        # Based on the concept of Nash equilibrium, the operation stops when no element exchange can reduce the
        # bias or when the bias is less than args.resource_bias_precision.
        while True:
            if cal_bias_resource(cpu_region_list, memory_region_list, cpu_region_list,
                        bandwidth_region_list) < args.resource_bias_precision:
                flag = 1
            if flag == 1:
                break
            flag = 1

            # Check the disk_region_list to find if there are swaps that reduce bias.
            for i in range(len(disk_region_list)):
                for j in range(len(disk_region_list) - i):
                    if pre_total_bias < args.resource_bias_precision:
                        break
                    tmp_disk = [one_data for one_data in disk_region_list]
                    tmp_value = tmp_disk[i]
                    tmp_disk[i] = tmp_disk[i + j]
                    tmp_disk[i + j] = tmp_value
                    cur_total_bias = cal_bias_resource(tmp_disk, memory_region_list, cpu_region_list,
                                              bandwidth_region_list)
                    if cur_total_bias < pre_total_bias:
                        pre_total_bias = cur_total_bias
                        disk_region_list = tmp_disk
                        flag = 0

            # Check the memory_region_list to find if there are swaps that reduce bias.
            for i in range(len(memory_region_list)):
                for j in range(len(memory_region_list) - i):
                    if pre_total_bias < args.resource_bias_precision:
                        break
                    tmp_memory = [one_data for one_data in memory_region_list]
                    tmp_value = tmp_memory[i]
                    tmp_memory[i] = tmp_memory[i + j]
                    tmp_memory[i + j] = tmp_value
                    cur_total_bias = cal_bias_resource(disk_region_list, tmp_memory, cpu_region_list,
                                              bandwidth_region_list)
                    if cur_total_bias < pre_total_bias:
                        pre_total_bias = cur_total_bias
                        memory_region_list = tmp_memory
                        flag = 0
            # Check the cpu_region_list to find if there are swaps that reduce bias.
            for i in range(len(cpu_region_list)):
                for j in range(len(cpu_region_list) - i):
                    if pre_total_bias < args.resource_bias_precision:
                        break
                    tmp_cpu = [one_data for one_data in cpu_region_list]
                    tmp_value = tmp_cpu[i]
                    tmp_cpu[i] = tmp_cpu[i + j]
                    tmp_cpu[i + j] = tmp_value
                    cur_total_bias = cal_bias_resource(disk_region_list, memory_region_list, tmp_cpu,
                                              bandwidth_region_list)
                    if cur_total_bias < pre_total_bias:
                        pre_total_bias = cur_total_bias
                        cpu_region_list = tmp_cpu
                        flag = 0

            # Check the bandwidth_region_list to find if there are swaps that reduce bias.
            for i in range(len(bandwidth_region_list)):
                for j in range(len(bandwidth_region_list) - i):
                    if pre_total_bias < args.resource_bias_precision:
                        break
                    tmp_bandwidth = [one_data for one_data in bandwidth_region_list]
                    tmp_value = tmp_bandwidth[i]
                    tmp_bandwidth[i] = tmp_bandwidth[i + j]
                    tmp_bandwidth[i + j] = tmp_value
                    cur_total_bias = cal_bias_resource(disk_region_list, memory_region_list, cpu_region_list,
                                              tmp_bandwidth)
                    if cur_total_bias < pre_total_bias:
                        pre_total_bias = cur_total_bias
                        bandwidth_region_list = tmp_bandwidth
                        flag = 0

        # Record each device attribute in the form of [disk, memory, cpu, bandwidth].
        device_resource_tmp = []
        for i in range(len(bandwidth_region_list)):
            device_resource_tmp.append([disk_region_list[i], memory_region_list[i], cpu_region_list[i],
                                        bandwidth_region_list[i]])
        device_resource_list[region] = device_resource_tmp
    print('End')
    return device_resource_list

    # step 5: The goal of this step is to take the geographic distribution of GDP/population and the edge server resource
    #         allocation in the region as input to output the geographic distribution of edge servers in this region.
def ESMG_step5(GDP_data, device_resource_list, geographic_data):
    print('Step 5 - start')
    cur_bias = cal_bias_geo(GDP_data, device_resource_list)
    pre_bias = cur_bias

    # Following the idea of Nash equilibrium, the run is stopped when no action can reduce the
    # bias or when the bias is less than args.geo_bias_precision.
    flag = 0
    while True:
        if flag == 1 or pre_bias<args.geo_bias_precision:
            break
        flag = 1
        for region in device_resource_list:
            tmp_value = device_resource_list[region]
            for device in range(len(device_resource_list[region])):
                for value in range(len(device_resource_list[region][device])):
                    device_resource_list[region][device][value] = device_resource_list[region][device][value] * 0.8
            cur_bias = cal_bias_geo(GDP_data, device_resource_list)
            if cur_bias < pre_bias:
                pre_bias = cur_bias
                flag = 0
            else:
                device_resource_list[region] = tmp_value

            tmp_value = device_resource_list[region]
            for device in range(len(device_resource_list[region])):
                for value in range(len(device_resource_list[region][device])):
                    device_resource_list[region][device][value] = device_resource_list[region][device][value] * 1.2
            cur_bias = cal_bias_geo(GDP_data, device_resource_list)
            if cur_bias < pre_bias:
                pre_bias = cur_bias
                flag = 0
            else:
                device_resource_list[region] = tmp_value

    all_device = []
    for region in device_resource_list:
        for device in range(len(device_resource_list[region])):
            one_device = device_resource_list[region][device]
            one_device.append(region)
            # Add the latitude
            one_device.append(random.uniform(geographic_data[region][0][0], geographic_data[region][0][1]))
            # Add the longitude
            one_device.append(
                random.uniform(geographic_data[region][1][0], geographic_data[region][1][1]))
            if int(one_device[0]/1024/1024/1024) * int(one_device[1]/1024/1024/1024) * int(one_device[2]) * int(one_device[3]/1024/1024) !=0:
                all_device.append(one_device)

    all_device_df = pd.DataFrame(
        {
            'Disk (GB)': [int(device[0]/1024/1024/1024) for device in all_device],
            'Memory (GB)': [int(device[1]/1024/1024/1024) for device in all_device],
            'CPU (Core)': [int(device[2]) for device in all_device],
            'Bandwidth (MB/s)': [int(device[3]/1024/1024) for device in all_device],
            'Region': [device[4] for device in all_device],
            'Latitude': [device[5] for device in all_device],
            'Longitude': [device[6] for device in all_device]
        })
    all_device_df.to_csv(args.devices_out_addr, encoding="utf-8-sig", mode="a", header=True, index=True)
    print('End')
    print('The number of successfully generated edge devices is: %d, which has been stored as file: %s.' % (len(all_device), args.devices_out_addr))
    return all_device_df

def ESMG(GDP_addr, geographic_addr):
    # Load the data
    GDP_data = load_json(GDP_addr)
    geographic_data = load_json(geographic_addr)

    # step 1: The goal of this step is to take the total GDP or population of a region as input to output the total amount
    #         of each type of resource for the whole region.
    total_cpu, total_memory, total_disk, total_bandwidth = ESMG_step1(GDP_data)

    # step 2: The goal of this step is to use the total amount of each type of resource in the region as input to output
    #         the number of edge servers in this region.
    device_num = ESMG_step2(total_cpu, total_memory, total_disk, total_bandwidth)

    # step 3: The goal of this step is to take the total number of each resource and the number of edge servers in the
    #         region as input to output the distribution of each resource in this region.
    disk_distribution_each_region, memory_distribution_each_region, cpu_distribution_each_region, bandwidth_distribution_each_region = ESMG_step3(
        total_cpu, total_memory, total_disk, total_bandwidth, device_num)

    # step 4: The goal of this step is to take the distribution of each resource in the region as input to output the
    #         resource configuration for each edge server in this region.
    device_resource_list = ESMG_step4(disk_distribution_each_region, memory_distribution_each_region,
                                      cpu_distribution_each_region, bandwidth_distribution_each_region)
    # step 5: The goal of this step is to take the geographic distribution of GDP/population and the edge server resource
    #         allocation in the region as input to output the geographic distribution of edge servers in this region.
    ESMG_step5(GDP_data, device_resource_list, geographic_data)

    return 0


if __name__ == '__main__':
    ESMG(args.GDP_addr, args.geographic_addr)
