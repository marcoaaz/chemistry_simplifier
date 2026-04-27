
'''
main_functions.py

Follows 'main_functions.py' in vsiFormatter/. 

Created: 19-Sep-2025, Marco Acevedo
Updated: 


'''
#Dependencies

import os
import psutil
import re
import glob

import pandas as pd
import torch

def get_device(): 
    # get the computation device as torch.device object
    
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

def natural_sort_key(s):
    """ Helper to sort strings containing numbers naturally """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def getFileList(imageFolder, input_expression, extension):
	#Examples:
	#pattern = re.compile(r".*/(.*)_(\d+)_(\d+)\.tiff") 
	#pattern = re.compile(r".+\\153874_(.+)-(.+)\.tiff") 
	#fileList = glob.glob(f"{imageFolder}/91702-*-*.tiff")   

	if input_expression == "":
		expression = "(.+)"
	else:
		expression = input_expression  
	extension2 = r"\." + extension

	#Note: fileList and pattern need to be similar
	fileList = glob.glob(os.path.join(imageFolder, f"*.{extension}"))
	fileList.sort(key=natural_sort_key)
	#print(np.transpose(np.array(fileList)))           

	# pattern = re.compile(r".+\\" + expression + extension2)        
	pattern = re.compile(r".+" + re.escape(os.sep) + expression + extension2)

	return fileList, pattern

def build_filePaths(fileList, pattern, destDir):

	list1 = []    
	for filename in fileList:

		match = pattern.match(filename)       
		
		if match:            
			element = match.group(1) #most common                    
			list1.append([element, filename])		            
		
	path_table = pd.DataFrame(list1)
	path_table.columns = ["parsed_element", "path"]

	#Save table
	file_name1 = os.path.join(destDir, "inputs_refresh.csv")
	path_table.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)
	
	return path_table


def qListWidget_list(list_widget, workingDir): #For GUI
		
	item_texts = []
	for i in range(list_widget.count()):
		item_texts.append(list_widget.item(i).text())

	chosen_table = pd.DataFrame(item_texts)
	chosen_table.columns = ["path"]

	#Save table
	file_name1 = os.path.join(workingDir, "inputs_run.csv")
	chosen_table.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)

	return chosen_table

def predict_datasetSize(tiles_metadata, n_layers_input, scale, fileSize_th_GB = 4, bitDepth = 32):
	#maximum allowed fraction of montage for training (prevents RAM overload) 	
	#prevents pytorch issue: OverflowError: cannot serialize a bytes object larger than 4 GB          

	#Predict file size	
	imageHeight = tiles_metadata.loc[tiles_metadata['x'] == 0, :]['H'].sum()
	imageWidth = tiles_metadata.loc[tiles_metadata['y'] == 0, :]['W'].sum()    		
	scale_factor = 1/scale	
		
	fileSize_GB = scale_factor*(imageHeight*imageWidth*n_layers_input*(bitDepth/8))/(10**9) #GB    

	if fileSize_GB > fileSize_th_GB:
		fraction = fileSize_th_GB/fileSize_GB    
	else:
		fraction = 1

	print(f"Sampled size: {fileSize_GB:.2f} GB | Max sampling fraction: {fraction:.4f}")

	return fraction

