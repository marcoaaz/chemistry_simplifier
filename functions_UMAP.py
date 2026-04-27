'''
functions_UMAP.py

Contains the functions for running the Uniform manifold approximation and projection within Chemistry Simplifier v1 executable.

Created: 19-Sep-2025, Marco Acevedo
Updated: 29-Sep-25, 3-Apr-26, M.A.

'''

import os    
import sys
import time
from tqdm import tqdm
from multiprocessing import Pool
import gc

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

worker_image_path = None #global variable

def init_worker(file_path):
	"""
	This runs ONCE per CPU core when the pool starts.
	"""	
	global worker_image_path #load locally (faster option)
	worker_image_path = file_path

	# global worker_image    
	# worker_image = pyvips.Image.new_from_file(file_path, access="random") #jumping around x, y	

def incremental_loading_UMAP(tiles_metadata, fraction, scaler_input, scaling_type, neighbors_input, min_dist_input, outputFolder):

	#Default
	n_channels = 3
	min_required = n_channels + 1 #>3 for first partial fit
	seed_value = 0 #seed for subsetting arrays in training/test sets            
	np.random.seed(seed_value)       

	#Load stack
	unique_file = tiles_metadata['filepath'].iloc[0]
	full_image = pyvips.Image.new_from_file(unique_file, access="random") #jumping around x, y

	#Load random sample of tile pixels 	
	tile_arrays = []
	for csv_row in tiles_metadata.itertuples():
   
		#Tile		
		image_temp = full_image.crop(csv_row.pixel_x, csv_row.pixel_y, csv_row.W, csv_row.H)		

		#To numpy matrix
		image_np = image_temp.numpy()
		h, w, bands = image_np.shape  
		
		#Vectorising
		image_xyz = image_np.reshape(-1, bands).astype(np.float32) 
		image_xyz = np.unique(image_xyz, axis=0) #exclude background pixels (no variance)

		del image_np

		#Subsampling				
		n_rows = image_xyz.shape[0]   		
		to_row = int(n_rows*fraction)

		#medicine: ensure working with enough rows
		if n_rows <= min_required:			
			continue				 		
		
		to_row = max(min_required, to_row)
		indices = np.random.permutation(n_rows) #scrambling
		sampling_idx = indices[:to_row]        					
		image_xyz2 = image_xyz[sampling_idx, :]#selected data	

		del image_xyz
		
		if scaling_type == 'centering':
			image_xyz3 = centering_operation(image_xyz2, scaler_input)
		elif scaling_type == 'standardising':
			image_xyz3 = standardising_operation(image_xyz2, scaler_input)		
	
		tile_arrays.append(image_xyz3) 		

		del image_xyz2		
		del image_xyz3 

	#Concatenate sub-samples
	dataset = np.concatenate([i for i in tile_arrays])	
	# print(dataset.shape)

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

def transform_tiles_umap(tiles_metadata, resolution, scaler_input, scaling_type, model_path, outputFolder2, n_cores):		
	
	umap_model = joblib.load(model_path)
	
	fileList = tiles_metadata['filepath']
	total_len = len(fileList) #number of iterations
	bigtiff_path = fileList.iloc[0] #montage

	#Transform each tile	
	args = ((csv_row, resolution, scaler_input, scaling_type, umap_model, outputFolder2)
		 for csv_row in tiles_metadata.to_dict('records'))		
	
	#incremental results (imap)        
	with Pool(
		processes=n_cores,
		initializer=init_worker, 
		initargs=(bigtiff_path,)		
		) as pool: 
		fileList2 = list( tqdm(
			pool.imap(_unpack_and_call, args, chunksize=1), 
			total= total_len, 
			ascii=True
			) )		

	return fileList2

def umap_section(csv_row, resolution, scaler_input, scaling_type, umap_model, outputFolder2):
	
	#Default
	n_channels = 3
	resolution_factor = 1/resolution

	#Tile	
	pixel_x = csv_row['pixel_x'] #start pixel
	pixel_y = csv_row['pixel_y']
	W = csv_row['W'] #current_w
	H = csv_row['H']
	x_val = str(csv_row['x']) #col_idx
	y_val = str(csv_row['y'])

	global worker_image_path	
	worker_image = pyvips.Image.new_from_file(worker_image_path, access="random")

	image_temp = worker_image.crop(pixel_x, pixel_y, W, H) #tile stack	

	#Downsampling
	if resolution_factor != 1:
		image_temp = image_temp.resize(resolution_factor)

	#To numpy matrix
	image_np = image_temp.numpy()
	h, w, bands = image_np.shape  
	
	image_xyz2 = image_np.reshape(-1, bands).astype(np.float32) #.astype(np.float32) 

	del image_temp 
	del worker_image #only when using worker_image_path
	del image_np

	#Normalisation
	if scaling_type == 'centering':
		image_xyz3 = centering_operation(image_xyz2, scaler_input) #original code
	elif scaling_type == 'standardising':
		image_xyz3 = standardising_operation(image_xyz2, scaler_input)	                    		

	#Transform
	image_features = umap_model.transform(image_xyz3)	
	
	tile_umap = image_features.reshape(h, w, n_channels).astype(np.float32) #pyvips 64-bit floating-point fails reload
	tile_umap2 = pyvips.Image.new_from_array(tile_umap)                            	

	#output file
	output_filename = f"{x_val}_{y_val}.tif"
	output_path = os.path.join(outputFolder2, output_filename)
	tile_umap2.write_to_file(output_path) 		
	
	del image_xyz3
	del tile_umap
	del tile_umap2	
	gc.collect() #trigger cleanup

	return output_path