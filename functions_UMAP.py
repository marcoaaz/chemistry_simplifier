'''
functions_UMAP.py

Contains the functions for running the Uniform manifold approximation and projection within Chemistry Simplifier v1 executable.

Created: 19-Sep-2025, Marco Acevedo
Updated: 29-Sep-25

'''

import os    
import sys
import time
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np

#LLVMLITE issue
add_dll_dir = getattr(os, 'add_dll_directory', None) #Windows=True
vipsbin2 = 'E:/Alienware_March 22/current work/00-new code May_22/dimReduction_v2/chemSimplifier3/Lib/site-packages/llvmlite/binding'
if getattr(sys, 'frozen', False):	# Running in a PyInstaller bundle		

	bundle_dir = os.path.abspath(os.path.dirname(__file__)) #relative path	
	vip_dlls = os.path.join(bundle_dir, 'llvmlite/binding')

else: # for regular Python environment
	vip_dlls = vipsbin2

if callable(add_dll_dir): 
	add_dll_dir(vip_dlls)
else:
	os.environ['PATH'] = os.pathsep.join((vip_dlls, os.environ['PATH']))

import umap
import joblib #for saving UMAP model
from functions_pyramids import centering_operation, standardising_operation

#VIPS
add_dll_dir = getattr(os, 'add_dll_directory', None) #Windows=True
vipsbin = 'c:/vips-dev-8.16/bin' #r'c:\vips-dev-8.16\bin'
if getattr(sys, 'frozen', False):	# Running in a PyInstaller bundle		

	bundle_dir = os.path.abspath(os.path.dirname(__file__)) #relative path	
	vip_dlls = os.path.join(bundle_dir, 'vips')

else: # for regular Python environment
	vip_dlls = vipsbin

if callable(add_dll_dir): 
	add_dll_dir(vip_dlls)
else:
	os.environ['PATH'] = os.pathsep.join((vip_dlls, os.environ['PATH']))

import pyvips
# print("vips version: " + str(pyvips.version(0))+"."+str(pyvips.version(1))+"."+str(pyvips.version(2)))


def incremental_loading_UMAP(fileList, scale, fraction, scaler_input, scaling_type, neighbors_input, min_dist_input, outputFolder):

	#Default
	n_channels = 3
	seed_value = 0 #seed for subsetting arrays in training/test sets            
	np.random.seed(seed_value)       
	scale_factor = 1/scale

	#Load random sample of tile pixels for UMAP
	
	tile_arrays = []
	for file in fileList:
   
		image_temp = pyvips.Image.new_from_file(file)		

		#Downsampling
		image_temp = image_temp.resize(scale_factor)

		#To numpy matrix
		image_np = image_temp.numpy()
		dim_shape = image_np.shape  
		image_xyz = np.reshape(image_np, (-1, dim_shape[2])) #39 channels
		
		#Subsampling
		n_rows = image_xyz.shape[0]                
		to_row = int(n_rows*(fraction))
		indices = np.random.permutation(n_rows) #scrambling
		sampling_idx = indices[:to_row]        
					
		image_xyz2 = image_xyz[sampling_idx, :]#selected data

		if scaling_type == 'centering':
			image_xyz3 = centering_operation(image_xyz2, scaler_input) #original code
		elif scaling_type == 'standardising':
			image_xyz3 = standardising_operation(image_xyz2, scaler_input)	

		tile_arrays.append(image_xyz3)    
    
    #Sub-sample
	dataset = np.concatenate([i for i in tile_arrays])	
	print(dataset.shape)

	start = time.time()
    #Fit UMAP
	umap_model = umap.UMAP(n_neighbors= neighbors_input, min_dist= min_dist_input,
                      metric='euclidean', n_components= n_channels,
                     ).fit(dataset) #euclidean, correlation, cosine
    
	end = time.time()
	print(f"Fitting UMAP took {(end-start)/60:.3} min")

    #Save model	
	model_path = os.path.join(outputFolder, "umap_model.xml")	
	joblib.dump(umap_model, model_path) #umap transform

	return dataset, model_path         

def _unpack_and_call(args):
		return umap_section(*args)

def transform_tiles_umap(fileList, resolution, scaler_input, scaling_type, model_path, outputFolder2, n_cores):		
	
	umap_model = joblib.load(model_path)
	
	#Transform each tile
	total_len = len(fileList)
	args = ((file, resolution, scaler_input, scaling_type, umap_model, outputFolder2)
		 for file in fileList)		
	
	with Pool(processes= n_cores) as pool: # imap for incremental results        
		fileList2 = list( tqdm(pool.imap(_unpack_and_call, args, chunksize=1), total= total_len, ascii=True) )
	
	# pool = multiprocessing.Pool(processes=n_cores)	
	# fileList2 = pool.starmap(umap_section, args)		

	return fileList2

def umap_section(file, resolution, scaler_input, scaling_type, umap_model, outputFolder2):
	#Default
	n_channels = 3
	resolution_factor = 1/resolution

	#Loop
	image_temp = pyvips.Image.new_from_file(file)

	#Downsampling
	image_temp = image_temp.resize(resolution_factor)

	#To numpy matrix
	image_np = image_temp.numpy()
	dim_shape = image_np.shape  
	image_xyz2 = np.reshape(image_np, (-1, dim_shape[2])) #39 channels  

	if scaling_type == 'centering':
		image_xyz3 = centering_operation(image_xyz2, scaler_input) #original code
	elif scaling_type == 'standardising':
		image_xyz3 = standardising_operation(image_xyz2, scaler_input)	                    		

	#Transform
	image_features = umap_model.transform(image_xyz3)
	
	tile_umap = np.reshape(image_features, (dim_shape[0], dim_shape[1], n_channels) )        
	
	tile_umap2 = pyvips.Image.new_from_array(tile_umap)                            
	tile_umap3 = tile_umap2.cast("float") #note: 64-bit floating-point fails reload

	#output file
	output_filename = os.path.basename(file)
	output_path = os.path.join(outputFolder2, output_filename)
	tile_umap3.write_to_file(output_path) 		
	
	return output_path