
import json
import re
import numpy as np
import math
import os
import csv

def load_dict_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, return an empty dictionary
        data = {}
    return data

# Add a new key-value pair to the dictionary
def add_new_instance_if_not_exists(file_path, key, value):
    # Load the dictionary
    data = load_dict_from_json(file_path)
    
    # Check if the key exists
    if key in data:
        print(f"Key '{key}' already exists in the dictionary with value: {data[key]}")
    else:
        # Add the new key-value pair if the key doesn't exist
        data[key] = value
        #print(f"Adding new key '{key}' with value: {value}")
        
        # Save the updated dictionary back to the JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)


# Function to search for a key in the dictionary
def search_instance_in_dict(dictionary, key):
    if key in dictionary:
        return True
    else:
        return False


def objectivePriority(latency, energy, area, accuracy, coeff_lat, coeff_en, coeff_ar, coeff_acc, typeObj, latency_max, energy_max, area_max, accuracy_max, latency_min=0.0, energy_min=0.0, area_min=0.0, accuracy_min=0.0):
            if typeObj=='e':
                obj=energy
            elif typeObj=='l':
                obj=latency
            elif typeObj=='a':
                obj=area
            elif typeObj=='ela':
                obj=latency*energy*area
            elif typeObj=='el': 
                obj=latency*energy
            elif typeObj=='e_acc':
                obj=energy/accuracy
            elif typeObj=='l_acc':
                obj=latency/accuracy
            elif typeObj=='a_acc':
                obj=area/accuracy
            elif typeObj=='ela_acc':
                obj=latency*energy*area/accuracy
            elif typeObj=='el_acc': 
                obj=latency*energy/accuracy
            # most of the time minimum values are jsut set to zero 
            elif typeObj=='ela_acc_wp': #weighted product 
                obj=pow(((latency-latency_min)/(latency_max-latency_min)), coeff_lat)*pow(((energy-energy_min)/(energy_max-energy_min)), coeff_en)*pow(((area-area_min)/(area_max-area_min)), coeff_ar)/pow(((accuracy-accuracy_min)/(accuracy_max-accuracy_min)), coeff_acc)
            elif typeObj=='ela_acc_ws': #weighted sum 
                obj=coeff_lat*((latency-latency_min)/(latency_max-latency_min))+coeff_en*((energy-energy_min)/(energy_max-energy_min))+coeff_ar*((area-area_min)/(area_max-area_min))-coeff_acc*((accuracy-accuracy_min)/(accuracy_max-accuracy_min))
                #obj=latency*energy*area/accuracy
            #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
            return obj

def objective(latency, energy, area, accuracy, coeff_lat, coeff_en, coeff_ar, coeff_acc, typeObj):
    if typeObj=='e':
        obj=energy
    elif typeObj=='l':
        obj=latency
    elif typeObj=='a':
        obj=area
    elif typeObj=='acc':
        obj=1./accuracy
    elif typeObj=='ela':
        obj=latency*energy*area
    elif typeObj=='el': 
        obj=latency*energy
    elif typeObj=='ea': 
        obj=latency*area
    elif typeObj=='la': 
        obj=latency*area
    elif typeObj=='e_acc':
        obj=energy/accuracy
    elif typeObj=='l_acc':
        obj=latency/accuracy
    elif typeObj=='a_acc':
        obj=area/accuracy
    elif typeObj=='ela_acc':
        obj=latency*energy*area/accuracy
    elif typeObj=='el_acc': 
        obj=latency*energy/accuracy
    #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
    return obj

def load_dict_from_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, return an empty dictionary
        data = {}
    return data

# Add a new key-value pair to the dictionary
def add_new_instance_if_not_exists(file_path, key, value):
    # Load the dictionary
    data = load_dict_from_json(file_path)
    
    # Check if the key exists
    if key in data:
        print(f"Key '{key}' already exists in the dictionary with value: {data[key]}")
    else:
        # Add the new key-value pair if the key doesn't exist
        data[key] = value
        #print(f"Adding new key '{key}' with value: {value}")
        
        # Save the updated dictionary back to the JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)


# Function to search for a key in the dictionary
def search_instance_in_dict(dictionary, key):
    if key in dictionary:
        return True
    else:
        return False

def get_dictCIMNAS(key):

    best_joint=key
    parts = best_joint.split('_')
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    #print(parts)
    numbers = [part for part in parts if number_pattern.match(part)]
    #print(numbers)
    hardware_param=numbers[:9]
    def convert_to_number(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    
    hardware_number_list = [convert_to_number(s) for s in hardware_param]
    # fi you want real buffer value
    #number_list[8]=round(int(numbers[8])*2048/1024/1024/8)
    #print(hardware_number_list)
    d_list = [int(char) for char in numbers[9]]
    #print(d_list)
    ks_list = [int(char) for char in numbers[10]]
    #print(ks_list)
    pw_w_bits_list=[int(char) for char in numbers[11]]
    #print(pw_w_bits_list)
    pw_a_bits_list=[int(char) for char in numbers[12]]
    #print(pw_a_bits_list)
    dw_w_bits_list=[int(char) for char in numbers[13]]
    #print(dw_w_bits_list)
    dw_a_bits_list=[int(char) for char in numbers[14]]
    #print(dw_a_bits_list)
    e0=[4.0, 4.5, 5.0, 5.5, 6.0]
    # 7 values
    e1_4=[4.0, 4.333333333333333, 4.666666666666667, 5.0, 5.333333333333333, 5.666666666666667, 6.0]
    # 11 values
    e5_8=[4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
    # 21 val
    e9_12=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
    # 25 val
    e13_16=[4.0, 4.083333333333333, 4.166666666666667, 4.25, 4.333333333333333, 4.416666666666667, 4.5, 4.583333333333333, 4.666666666666667, 4.75, 4.833333333333333, 4.916666666666667, 5.0, 5.083333333333333, 5.166666666666667, 5.25, 5.333333333333333, 5.416666666666667, 5.5, 5.583333333333333, 5.666666666666667, 5.75, 5.833333333333333, 5.916666666666667, 6.0]
    # 49 val
    e17_20=[4.0, 4.041666666666667, 4.083333333333333, 4.125, 4.166666666666667, 4.208333333333333, 4.25, 4.291666666666667, 4.333333333333333, 4.375, 4.416666666666667, 4.458333333333333, 4.5, 4.541666666666667, 4.583333333333333, 4.625, 4.666666666666667, 4.708333333333333, 4.75, 4.791666666666667, 4.833333333333333, 4.875, 4.916666666666667, 4.958333333333333, 5.0, 5.041666666666667, 5.083333333333333, 5.125, 5.166666666666667, 5.208333333333333, 5.25, 5.291666666666667, 5.333333333333333, 5.375, 5.416666666666667, 5.458333333333333, 5.5, 5.541666666666667, 5.583333333333333, 5.625, 5.666666666666667, 5.708333333333333, 5.75, 5.791666666666667, 5.833333333333333, 5.875, 5.916666666666667, 5.958333333333333, 6.0]
    e_list=[int(s) for s in numbers[-21:]]
    e_list_index=e_list[:]
    #print(e_list)
    e_list[0]=e0[e_list[0]]
    e_list[1:5]=[e1_4[s] for s in e_list[1:5]]
    e_list[5:9]=[e5_8[s] for s in e_list[5:9]]
    e_list[9:13]=[e9_12[s] for s in e_list[9:13]]
    e_list[13:17]=[e13_16[s] for s in e_list[13:17]]
    e_list[17:21]=[e17_20[s] for s in e_list[17:21]]
    #print(e_list)
    
    return hardware_number_list, d_list, ks_list, e_list, pw_w_bits_list, pw_a_bits_list, dw_w_bits_list, dw_a_bits_list,e_list_index

def findindexlist(e_list):
    e_list_index=e_list
    e0=[4.0, 4.5, 5.0, 5.5, 6.0]
    # 7 values
    e1_4=[4.0, 4.333333333333333, 4.666666666666667, 5.0, 5.333333333333333, 5.666666666666667, 6.0]
    # 11 values
    e5_8=[4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
    # 21 val
    e9_12=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
    # 25 val
    e13_16=[4.0, 4.083333333333333, 4.166666666666667, 4.25, 4.333333333333333, 4.416666666666667, 4.5, 4.583333333333333, 4.666666666666667, 4.75, 4.833333333333333, 4.916666666666667, 5.0, 5.083333333333333, 5.166666666666667, 5.25, 5.333333333333333, 5.416666666666667, 5.5, 5.583333333333333, 5.666666666666667, 5.75, 5.833333333333333, 5.916666666666667, 6.0]
    # 49 val
    e17_20=[4.0, 4.041666666666667, 4.083333333333333, 4.125, 4.166666666666667, 4.208333333333333, 4.25, 4.291666666666667, 4.333333333333333, 4.375, 4.416666666666667, 4.458333333333333, 4.5, 4.541666666666667, 4.583333333333333, 4.625, 4.666666666666667, 4.708333333333333, 4.75, 4.791666666666667, 4.833333333333333, 4.875, 4.916666666666667, 4.958333333333333, 5.0, 5.041666666666667, 5.083333333333333, 5.125, 5.166666666666667, 5.208333333333333, 5.25, 5.291666666666667, 5.333333333333333, 5.375, 5.416666666666667, 5.458333333333333, 5.5, 5.541666666666667, 5.583333333333333, 5.625, 5.666666666666667, 5.708333333333333, 5.75, 5.791666666666667, 5.833333333333333, 5.875, 5.916666666666667, 5.958333333333333, 6.0]
    e_list_index[0]=e0.index(e_list[0])
    e_list_index[1:5]=[e1_4.index(s) for s in e_list[1:5]]
    e_list_index[5:9]=[e5_8.index(s) for s in e_list[5:9]]
    e_list_index[9:13]=[e9_12.index(s) for s in e_list[9:13]]
    e_list_index[13:17]=[e13_16.index(s) for s in e_list[13:17]]
    e_list_index[17:21]=[e17_20.index(s) for s in e_list[17:21]]
    return e_list_index

