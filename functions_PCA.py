'''
functions_PCA.py

Contains the functions for running Principal Component Analysis within Chemistry Simplifier v1 executable.

Created: 19-Sep-2025, Marco Acevedo
Updated: 29-Sep-25

Documentation:
resizing interpolation used: https://mazzo.li/posts/lanczos.html


'''
import os    
import sys
import pickle

import numpy as np
import pandas as pd   
from sklearn.decomposition import IncrementalPCA

import multiprocessing

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


def incremental_PCA(fileList, scale, fraction, scaler_input, outputFolder):

	#Default
	n_channels = 3
	seed_value = 0 #seed for subsetting arrays in training/test sets            
	np.random.seed(seed_value)       
	scale_factor = 1/scale
   
	#Fit incremental PCA on sampled tiles    
	ipca = IncrementalPCA(n_components= n_channels)
	
	for file in fileList:
   
		image_temp = pyvips.Image.new_from_file(file)		

		#Downsampling
		image_temp = image_temp.resize(scale_factor) #kernel=pyvips.Kernel.LANCZOS3 default

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

		#Standardising (z-score)
		n_rows2 = image_xyz2.shape[0]
		mean_temp = scaler_input.iloc[:, 0].to_numpy().reshape((1, -1)) #row		
		std_temp = scaler_input.iloc[:, 1].to_numpy().reshape((1, -1))				
		mean_temp2 = np.tile(mean_temp, (n_rows2, 1) )						
		image_xyz3 = (image_xyz2 - mean_temp2) / std_temp #broadcasting

		ipca.partial_fit(image_xyz3)    

	#Save model
	model_path = os.path.join(outputFolder, 'pca_model.pkl')
	with open(model_path, 'wb') as file_model:
		pickle.dump(ipca, file_model)

	return model_path           

def transform_tiles_pca(fileList, resolution, scaler_input, model_path, outputFolder2, n_cores):	

	with open(model_path, 'rb') as file:
		ipca = pickle.load(file)    

	#Transform each tile
	args = ((file, resolution, scaler_input, ipca, outputFolder2)
		 for file in fileList)		
	
	pool = multiprocessing.Pool(processes=n_cores)	
	fileList2 = pool.starmap(pca_section, args)	

	return fileList2

def pca_section(file, resolution, scaler_input, ipca, outputFolder2):
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

	#Standardising (z-score)
	n_rows2 = image_xyz2.shape[0]
	mean_temp = scaler_input.iloc[:, 0].to_numpy().reshape((1, -1)) #row		
	std_temp = scaler_input.iloc[:, 1].to_numpy().reshape((1, -1))				
	mean_temp2 = np.tile(mean_temp, (n_rows2, 1) )						
	image_xyz3 = (image_xyz2 - mean_temp2) / std_temp #broadcasting

	#Transform
	image_features = ipca.transform(image_xyz3)
	
	tile_pca = np.reshape(image_features, (dim_shape[0], dim_shape[1], n_channels) )        
	
	tile_pca2 = pyvips.Image.new_from_array(tile_pca)                            
	tile_pca3 = tile_pca2.cast("float") #note: 64-bit floating-point fails reload

	#output file
	output_filename = os.path.basename(file)
	output_path = os.path.join(outputFolder2, output_filename)
	tile_pca3.write_to_file(output_path) 		
	
	return output_path