
'''
main_functions.py

Follows 'main_functions.py' in vsiFormatter/. 

Created: 19-Sep-2025, Marco Acevedo
Updated: 


'''
#Dependencies

import os
import psutil
import torch

def get_device(): # get the computation device    
    
    condition = torch.cuda.is_available()     
    if condition:        
        device1 = torch.device('cuda:0') #default: first GPU           
    else:        
        device1 = torch.device('cpu')   

    print(f"Using {device1}..")

    #Additional Info when using cuda
    if device1.type == 'cuda':

        n_devices = torch.cuda.device_count()
        curr_device = torch.cuda.current_device() #tell if its too old        
        device_name = torch.cuda.get_device_name(curr_device) #'GeForce GTX 950M'            
        # device_item = torch.cuda.device(curr_device)
        
        print(f'There are {n_devices} GPUs')
        print(device_name)
        # print('Memory Usage:')
        # print('Allocated:', round(torch.cuda.memory_allocated(curr_device)/1024**3,1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(curr_device)/1024**3,1), 'GB')   

    return device1       

def parse_system_info():

	#number of cores
	available_cores = os.cpu_count()

	#RAM tuple
	svmem = psutil.virtual_memory()
	total_RAM = get_size(svmem.total) #'31.66GB'
	available_RAM = get_size(svmem.available) 

	return available_cores, total_RAM, available_RAM

def get_size(bytes, suffix=""): #suffix="B"

	"""
	Scale bytes to its proper format
	e.g:
		1253656 => '1.20MB'
		1253656678 => '1.17GB'
	"""
	
	RAM_percentage = 75 #for JVM

	factor = 1024
	for unit in ["", "K", "M", "G", "T", "P"]:
		if bytes < factor:
			bytes2 = bytes*(RAM_percentage/100)

			return1 = f"{bytes:.0f}{unit}{suffix}"
			return2 = f"{bytes2:.0f}{unit}{suffix}"

			return return1, return2 #bytes:.2f
		bytes /= factor

