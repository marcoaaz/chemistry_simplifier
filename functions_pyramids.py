# -*- coding: utf-8 -*-
"""
functions_pyramids.py

Script useful for tiling and stacking chemical layers of very large (WSI) X-ray maps.

Citation: This is the update of the first half of the first script for dimensionality reduction 
'stackingCPS_synchrotron_v6.m’ in Acevedo Zamora et al. (2024). https://doi.org/10.1016/j.chemgeo.2024.121997

Current version: python 3.13.7, chemSimplifier2
Written by: Marco Andres, ACEVEDO ZAMORA

Updated:
Published (first version): 14-Nov-23
second version for whole-slide imaging: 19-Jun-24, 5-Sep-24
third version for GUI: 16-sep-25, 30-Sep-25, 9-Oct-25, 3-Apr-26

Follows previous work: tilingAndStacking_v5.py
Autoencoder scripts: python 3.7.9 (DSA_env)
Pyvips scripts: python 3.9.13 (vsi_trial1)
Current: python 3.9.13 (chemSimplifier3)

Documentation:
https://www.libvips.org/API/current/libvips-arithmetic.html#vips-stats
https://libvips.github.io/pyvips/vimage.html
https://www.libvips.org/API/current/Examples.html
https://github.com/libvips/libvips/issues/2600   
https://forum.image.sc/t/reading-regions-of-tif-files-with-more-than-3-channels/93299/6
https://forum.image.sc/t/pyvips-2-2-is-out-with-improved-numpy-and-pil-integration/66664/3

For VIPS setup:
Initially, download VIPS and unpack in C: https://github.com/libvips/build-win64-mxe/releases/tag/v8.16.0
Then, in console run: conda config --set auto_activate_base false

"""
#!/usr/bin/python3

#Dependencies   
import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #to provide colourmap options

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

from helperFunctions.mkdir_options import mkdir1, mkdir2, remove
from concurrent.futures import ThreadPoolExecutor #passes pointers not handles

from PIL import Image #to remove pyvips metadata
Image.MAX_IMAGE_PIXELS = None #trusting they are images

## Functions

#region Helper functions

def build_lut(cmap_name='jet'):
	#For recoloured (false-colour) images colormap	

	"""
	Uses Matplotlib colormaps to create a pyvips LUT.
	e.g.: 'jet', 'hot', 'viridis', 'magma', 'plasma', 'inferno'

	pyvips only (similar to 'jet'):
	# img_indexes = pyvips.Image.identity()
	# lut = img_indexes.falsecolour() #standard heatmap
	# #256x1 uchar, 3 bands, srgb, pngload	
	"""

	# 1. Get the colormap from matplotlib
	cmap = plt.get_cmap(cmap_name)
	
	# 2. Create 256 colors (R, G, B), drop alpha    
	colors = cmap(np.linspace(0, 1, 256))[:, :3] 
	
	# 3. Convert to 0-255 uchar format
	colors_8bit = (colors * 255).astype(np.uint8)
	
	# 4. Reshape for pyvips: must be 256x1 pixels with 3 bands    
	lut = pyvips.Image.new_from_memory(
		colors_8bit.tobytes(), 
		256, 1, 3, 
		'uchar'
		)

	return lut

def descriptive_colnames(type_list):
	
	stats_cols = ['min', 'max', 'sum', 'sumOfSquares', 'mean', 'stddev', 
						'min_x_coord', 'min_y_coord', 'max_x_coord', 'max_y_coord'] #first   
	threshold_cols = ["th_low_input", "th_high_input"] #second
	
	#first
	type_list2 = [output_type for output_type in type_list if output_type != "original"]
	type_list3 = ["original", *type_list2]
	stats_cols2 = [output_type + "_" + stat_type for output_type in type_list3 for stat_type in stats_cols ]
	#second
	threshold_cols2 = [output_type + "_" + threshold_type for output_type in type_list2 for threshold_type in threshold_cols ]

	descriptive_cols = stats_cols2 + threshold_cols2

	return descriptive_cols

#endregion

#region Image processing 

def clear_all_metadata(imgname): 	
	#Remove ImageJ pixel calibration that conflicts with registration (ImageJ BigWarp) 
	# referenced to the 'moving' image control points. 
	# Following Muhammad Abdullahi: 
	# https://thepythoncode.com/article/how-to-clear-image-metadata-in-python	
	
	with Image.open(imgname) as img: #avoid memory leaks
				
		# Create a new image with the same mode and size but without metadata.
		# without 'info' dict (metadata)
		img_without_metadata = Image.new(img.mode, img.size)
		img_without_metadata.paste(img)
		img_without_metadata.save(imgname)
	
	#Old:
	# img = Image.open(imgname)		
	# data = list(img.getdata()) #pixels without metadata		# 
	# img_without_metadata = Image.new(img.mode, img.size)
	# img_without_metadata.putdata(data)	
	# img_without_metadata.save(imgname)
	
def save_recoloured_channel(image_med, lut, destFile1):

	image_recoloured = image_med.maplut(lut)
	image_recoloured.write_to_file(destFile1)

#Extract from ray_tracing_module.py 
def channel_uint8(image_rs3):
	image_rs4 = image_rs3.cast("uchar") #uint8    

	return image_rs4

def find_P_thresholds(input_channel, percentOut, bit_precision = 16):
	#bit_precision: 8-bit=255; 16-bit=65535	
	
	calc_depth = (2**bit_precision) -1  
	stats = input_channel.stats() #eager pass (load data once)
	min_val = stats(0, 0)[0] 
	max_val = stats(1, 0)[0]    
	 
	#medicine 1: zero division if weight-decay is too low and no. epochs too high
	range_val = max((max_val - min_val), 0.1) 
	ratio = (range_val/calc_depth)

	#medicine 2: 'gint' is invalid or out of range
	p_low = percentOut + 0.005 
	p_high = 100 - p_low

	#Finding percentiles (16-bit), 'uchar'=8-bit, 'uint' or 'ushort'=16-bit  	
	image_rs2 = ((input_channel - min_val) / ratio).cast("ushort") #uint	

	th_low = image_rs2.percent(p_low) #'int'	
	th_high = image_rs2.percent(p_high)        	
	
	th_low_input = th_low*ratio + min_val 
	th_high_input = th_high*ratio + min_val                    

	#medicine 3: zero division when processing an artefact image (e.g., Synchrotron XFM Flux0)
	if th_low_input == th_high_input:
		th_high_input = th_high_input + 1

	return th_low_input, th_high_input

def channel_rescaled(input_channel, min_val, max_val, th_low_input, th_high_input):  
	#Min-Max Scaling (normalization)
	#Note: Use when the distribution is not normal and you need to preserve data relationships.
	#Following: https://au.mathworks.com/help/matlab/ref/rescale.html
		
	#Capping
	input_channel = (input_channel > th_high_input).ifthenelse(th_high_input, input_channel) #true, false
	input_channel = (input_channel < th_low_input).ifthenelse(th_low_input, input_channel)

	#Rescaling
	output_channel = min_val + (input_channel - th_low_input) * ( (max_val - min_val) / (th_high_input - th_low_input) ) 				

	return output_channel   


def channel_standardise(input_channel, mean_temp, std_temp):	
	output_channel = (input_channel - mean_temp) / std_temp #z-score		

	return output_channel  

def standardising_operation(image_xyz2, scaler_input):
	#Z-score Normalisation (standardisation).
	#Unitless method used in distance-based algorithms assuming a normal distribution 
	#Example: KNN, PCA, Logistic Regression, Linear Regression.
	#This is a NaN propagating function

	n_rows2 = image_xyz2.shape[0]	
	mean_temp = scaler_input.iloc[:, 0].to_numpy().reshape((1, -1)) #row		
	mean_temp2 = np.tile(mean_temp, (n_rows2, 1) )		
	std_temp = scaler_input.iloc[:, 1].to_numpy().reshape((1, -1))				
	image_xyz3 = (image_xyz2 - mean_temp2) / std_temp #broadcasting

	return image_xyz3

def centering_operation(image_xyz2, scaler_input):
	#Centering around zero. Preserve physical meaning and relative variance (same units)
	#Used in regression with polynomials (reduce collinearity)
	#This is a NaN propagating function

	n_rows2 = image_xyz2.shape[0]
	mean_temp = scaler_input.iloc[:, 0].to_numpy().reshape((1, -1)) #row		
	mean_temp2 = np.tile(mean_temp, (n_rows2, 1) )			
	image_xyz3 = (image_xyz2 - mean_temp2)

	return image_xyz3

#Similar to vsiFormatter > ray_tracing_module.py
def img_rescaled(image_cropped, percentOut):	
	
	#Descr. statistics (follows 'tilingAndStacking_v3.py')
	# minimum, maximum, sum, sum of squares, mean, standard deviation, 
	# x coordinate of minimum, y coordinate of minimum, 
	# x coordinate of maximum, y coordinate of maximum  	

	#Default
	min_val = 0
	max_val = 255	
	target_W = 5000

	#Generate thumbnail
	n_bands = image_cropped.bands	
	source_W = image_cropped.width
	ratio = target_W/source_W
	image_thumbnail = image_cropped.resize(ratio, kernel=pyvips.Kernel.NEAREST)
	
	out = pyvips.Image.stats(image_thumbnail)
	out1 = out.numpy()   	
	# > 6GB (74Kx45K montage) warning: vips_tracked: out of memory -- size == 3MB	
	
	#Preparing pyvips metadata table    	
	channel_list = [f"ch_{number:02d}" for number in range(n_bands)]
	stats_cols = ['min', 'max', 'sum', 'sumOfSquares', 'mean', 'stddev', 
						'min_x_coord', 'min_y_coord', 'max_x_coord', 'max_y_coord'] #first   
	threshold_cols = ["th_low_input", "th_high_input"] #second	
	
	stats_cols2 = [output_type + "_" + stat_type for output_type in channel_list for stat_type in stats_cols ]	
	threshold_cols2 = [output_type + "_" + threshold_type for output_type in channel_list for threshold_type in threshold_cols ]

	descriptive_cols = stats_cols2 + threshold_cols2
	
	#Load montage channels
	channel_list_out = [] #for images	
	item = [] #stats table row
	item_th = [] #th table row
	stats_list = []
	for i in range(n_bands):		
		
		statistic_vals = out1[i+1, :] #channel stats, row 0 has statistics for all bands together		

		#Direct indexing
		thumbnail_temp = image_thumbnail[i] #faster computation
		channel_temp = image_cropped[i]				

		#Positive, rescaled, capped and uint8 (useful for std, maxIndex, minIndex, PCA)
		thumbnail_positive = thumbnail_temp - statistic_vals[0]
		channel_positive = channel_temp - statistic_vals[0]				

		th_low_input, th_high_input = find_P_thresholds(thumbnail_positive, percentOut, 16)		
		channel_positive2 = channel_rescaled(channel_positive, min_val, max_val, th_low_input, th_high_input)		
		
		channel_list_out.append(channel_positive2)

		#Descriptive statistics	
		item.extend(statistic_vals) #linear/log statistics

		#Histogram thresholds  
		th_array = [th_low_input, th_high_input]
		item_th.extend(th_array)  

	item1 = item + item_th #add histogram thresholds            
	stats_list.append(item1)
	
	#Build info table
	stats_list2 = np.array(stats_list)                
	stats_list3 = pd.DataFrame(stats_list2)            
	stats_list3.columns = descriptive_cols  	

	#RGB
	image_rescaled = channel_list_out[0].bandjoin(channel_list_out[1:])

	return image_rescaled, stats_list3

def img_rescaled_prior(image_cropped, descriptiveRow):	
	#Descr. statistics (follows 'tilingAndStacking_v3.py')
	# minimum, maximum, sum, sum of squares, mean, standard deviation, 
	# x coordinate of minimum, y coordinate of minimum, x coordinate of maximum, y coordinate of maximum  	

	#Default
	min_val = 0
	max_val = 255	
	target_W = 5000

	#Generate thumbnail
	n_bands = image_cropped.bands	
	source_W = image_cropped.width
	ratio = target_W/source_W
	image_thumbnail = image_cropped.resize(ratio, kernel=pyvips.Kernel.NEAREST)
	
	out = pyvips.Image.stats(image_thumbnail)
	out1 = out.numpy()   	
	# > 6GB (74Kx45K montage) warning: vips_tracked: out of memory -- size == 3MB	
	
	#Preparing pyvips metadata table    	
	channel_list = [f"ch_{number:02d}" for number in range(n_bands)]
	stats_cols = ['min', 'max', 'sum', 'sumOfSquares', 'mean', 'stddev', 
						'min_x_coord', 'min_y_coord', 'max_x_coord', 'max_y_coord'] #first   
	threshold_cols = ["th_low_input", "th_high_input"] #second	
	
	stats_cols2 = [output_type + "_" + stat_type for output_type in channel_list for stat_type in stats_cols ]	
	threshold_cols2 = [output_type + "_" + threshold_type for output_type in channel_list for threshold_type in threshold_cols ]

	descriptive_cols = stats_cols2 + threshold_cols2
	
	#Load montage channels
	channel_list_out = [] #for images	
	item = [] #stats table row
	item_th = [] #th table row
	for i in range(n_bands):		
		
		statistic_vals = out1[i+1, :] #channel stats, row 0 has statistics for all bands together		

		#Use prior limits
		col1 = f"ch_{i:02d}_min"
		col2 = f"ch_{i:02d}_th_low_input"
		col3 = f"ch_{i:02d}_th_high_input"		
		min_temp = descriptiveRow[col1].item()
		th_low_input = descriptiveRow[col2].item()
		th_high_input = descriptiveRow[col3].item()

		#Process
		channel_temp = image_cropped[i] #direct indexing						
		channel_positive = channel_temp - min_temp #positive			
		channel_positive2 = channel_rescaled(channel_positive, min_val, max_val, th_low_input, th_high_input)					

		channel_list_out.append(channel_positive2)

		#Descriptive statistics
		item.extend(statistic_vals) #linear/log statistics

		#Histogram thresholds  
		th_array = [th_low_input, th_high_input]
		item_th.extend(th_array)  

	item1 = item + item_th #add histogram thresholds            
	
	#Build info table
	stats_list2 = np.array(item1).reshape(1, -1)                
	stats_list3 = pd.DataFrame(stats_list2)   
	stats_list3.columns = descriptive_cols 

	#RGB
	image_rescaled = channel_list_out[0].bandjoin(channel_list_out[1:])

	return image_rescaled, stats_list3

#endregion

#region Stitching
def stitch_crop_rescale(fileList2, tiles_across, outPct, outputFolder):    

	#identify method
	pattern = re.compile(r".*\\(\w+)_tiles")
	str1 = os.path.dirname(fileList2[0])
	match = pattern.match(str1)
	str2 = match.group(1)

	#Stitching
	tiles = []
	for file_pca in fileList2:
		
		pca_tile = pyvips.Image.new_from_file(file_pca)
		tiles.append(pca_tile)
	
	image_stitched = pyvips.Image.arrayjoin(tiles, across=tiles_across)        
	
	#crop background borders (optional)
	left, top, width, height = image_stitched.find_trim(threshold=0.001, background=[0])
	image_cropped = image_stitched.crop(left, top, width, height) #modify accordingly    
		
	#rescale bit-depth		
	image_rescaled, descriptiveStats = img_rescaled(image_cropped, outPct) #0-255 float
	image_rescaled_int = image_rescaled.cast("uchar") #uint8  
	image_rescaled_int2 = image_rescaled_int.copy()	#xres (inmutable)

	#Output montage	
	file_output = os.path.join(outputFolder, 'montage_' + str2 + '.tif')
	file_output2 = os.path.join(outputFolder, 'montage_' + str2 + '_8bit.tif')

	image_rescaled.write_to_file(file_output)
	image_rescaled_int2.write_to_file(file_output2) #resunit= cm, inch (mandatory)	
	clear_all_metadata(file_output2) #medicine
	
	#Output stats (for reproducibility)	
	file_output_table = os.path.join(outputFolder, "descriptiveStats_" + str2 + ".csv")	
	descriptiveStats.to_csv(file_output_table, sep=',', encoding='utf-8', index=False, header=True)

	return file_output2

def stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder):    

	#identify method
	pattern = re.compile(r".*\\(\w+)_tiles")
	str1 = os.path.dirname(fileList2[0])
	match = pattern.match(str1)
	str2 = match.group(1) #pca, dsa, umap

	#Stitching
	tiles = []
	for file_pca in fileList2:
		
		pca_tile = pyvips.Image.new_from_file(file_pca)
		tiles.append(pca_tile)
	
	image_stitched = pyvips.Image.arrayjoin(tiles, across=tiles_across)        
	
	#crop background borders (optional)
	left, top, width, height = image_stitched.find_trim(threshold=0.001, background=[0])
	image_cropped = image_stitched.crop(left, top, width, height) #modify accordingly    

	#rescale bit-depth (issue to fix in next version)	
	try:
		path1 = os.path.dirname(modelPATH)
		statsFile = os.path.join(path1, "descriptiveStats_" + str2 + ".csv")#chosen_table must match
		descriptiveRow = pd.read_csv(statsFile) #input stats	
		image_rescaled, descriptiveStats = img_rescaled_prior(image_cropped, descriptiveRow) #float
	except:
		#if previous process failed
		outPct = 0.2 #default (issue: should be retrieved from previos)
		image_rescaled, descriptiveStats = img_rescaled(image_cropped, outPct) #0-255 float
	
	image_rescaled_int = image_rescaled.cast("uchar") #uint8  
	image_rescaled_int2 = image_rescaled_int.copy()	#xres (inmutable)

	#Output montage	
	file_output = os.path.join(outputFolder, 'montage_' + str2 + '.tif')
	file_output2 = os.path.join(outputFolder, 'montage_' + str2 + '_8bit.tif')

	image_rescaled.write_to_file(file_output) 
	image_rescaled_int2.write_to_file(file_output2)	#resunit= cm, inch (mandatory)	
	clear_all_metadata(file_output2) #medicine

	#Output stats (for reproducibility)	
	file_output_table = os.path.join(outputFolder, "descriptiveStats_" + str2 + "_prior.csv")	
	descriptiveStats.to_csv(file_output_table, sep=',', encoding='utf-8', index=False, header=True)

	return file_output2

#endregion

#region Save stack image pyramids as BigTIFF

def generate_tile_config(type_list, img_w, img_h, tileSize, destDir):
	all_rows = []

	for type in type_list:		
		
		tiff_path = os.path.join(destDir, 'pyramids', f"{type}_stack.tif")
		
		if not os.path.exists(tiff_path):
			print(f"Warning: {tiff_path} not found. Skipping...")
			continue        

		# Calculate grid (pythonic order)
		for row_idx, y in enumerate(range(0, img_h, tileSize)):
			for col_idx, x in enumerate(range(0, img_w, tileSize)):
				current_w = min(tileSize, img_w - x)
				current_h = min(tileSize, img_h - y)
				
				all_rows.append({
					'filepath': tiff_path,
					'type': type,
					'x': col_idx, 
					'y': row_idx,
					'W': current_w,
					'H': current_h,
					'pixel_x': x, # Start pixel X
					'pixel_y': y  # Start pixel Y
				})

	# Create DataFrame
	tile_table = pd.DataFrame(all_rows)	
	tile_table2 = tile_table.sort_values(['type', 'y', 'x'], ascending=[True, True, True])

	# Save CSV
	file_name1 = os.path.join(destDir, "tileConfiguration.csv")
	tile_table2.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)

	# Console Summary
	for t in type_list:
		subset = tile_table2[tile_table2['type'] == t]

		if not subset.empty:
			tiles_down = subset['y'].max() + 1
			tiles_across = subset['x'].max() + 1
			print(f"Type '{t}': {tiles_down}x{tiles_across} tiles mapped.")

	return tile_table2

def save_stack(path_list, tileSize, destDir):
	#Saving original stack (used for 'ROIimageAnalysis_v7_wsi.m' script)

	destDir4 = os.path.join(destDir, 'pyramids')
	destDir4_files = destDir4 + '_files'
	try:
		remove(destDir4_files) #and subfolders
	except:
		print("Saving original images as Deep Zoom pyramid with only level 0.")

	#Loading tiles
	pages = [pyvips.Image.new_from_file(path) for path in path_list]
	image_stack = pages[0].bandjoin(pages[1:])	  

	image_stack.dzsave(destDir4, suffix='.tif', 
					skip_blanks=-1, background=0, 
					depth='one', overlap=0, tile_size= tileSize, 
					layout='dz') #Tile overlap in pixels*2


def save_transformed_stack_bigtiff(stack_layers_log, type, tileSize, destDir):
	
	# Setup naming
	destDir3 = os.path.join(destDir, 'pyramids')
	os.makedirs(destDir3, exist_ok=True)
	
	tiff_path = os.path.join(destDir3, f"{type}_stack.tif")

	# Transformed data
	image_stack = stack_layers_log[0].bandjoin(stack_layers_log[1:])    	
	
	# Save as Tiled BigTIFF
	print(f"Writing BigTIFF for {type}...")
	image_final = image_stack.copy(interpretation="multiband") 
	#Note: forced (interleave channels live next to each other on disk)
	
	image_final.tiffsave(tiff_path, 
						bigtiff=True, 
						tile=True, 
						tile_width = tileSize,
						tile_height = tileSize,
						compression="none",
						predictor="none") 	

	return tiff_path

def save_recoloured_stack(fileList0, type_list, linear_list, log_list, lut, destDir2, max_workers=4):
	"""
	Recoloured images for retrospective feedback (discarding artifact images)
	"""  
	print('Saving recoloured maps..')
	
	def process_recolour(i, filename):
		basename = os.path.splitext(os.path.basename(filename))[0]
		created_paths = []

		for t_type in type_list:
			# Select the correct list based on type
			if t_type == 'linear':
				temp_list = linear_list
			elif t_type == 'log':
				temp_list = log_list
			else:
				continue

			if i < len(temp_list):
				destFile1 = os.path.join(destDir2, f"{basename}_{t_type}.tif")                
				
				image_med = temp_list[i]
				image_med_uchar = (255 * image_med).cast("uchar")
				save_recoloured_channel(image_med_uchar, lut, destFile1)

				created_paths.append(destFile1)
		
		return created_paths

	# Execute in parallel
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		
		results = list(executor.map(lambda p: process_recolour(*p), enumerate(fileList0)))

		#Notes: 
		#results is a list of lists: [[path1, path2], [path3, path4], ...]
		#executor.map PRESERVES ORDER based on fileList0
		#list() to force the generator to execute
	
	dest_list = [path for sublist in results for path in sublist] #flattening	
	print(f'Finished saving {len(dest_list)} recoloured maps.')

	return dest_list

#endregion

#region Build image pyramids

def build_pyramids(chosen_table, scale, flip_ud, type_list, pctOut, filterSize, 
				   save_recoloured, lut, tileSize, destDir):

	#Default
	target_W = 5000 #same as 'img_rescaled'
	scale_factor = 1/scale	
	descriptive_cols = descriptive_colnames(type_list) #pyvips stats table    		

	#Load montages
	fileList0 = chosen_table["path"]

	log_list = [] #for images
	linear_list = []        
	stats_list = [] #for csv
	for filename in fileList0:

		basename_ext = os.path.basename(filename)		
		print(f"Loading {basename_ext}")                        
		
		#Load image
		full_image = pyvips.Image.new_from_file(filename) 
		img_w = full_image.width			
		n_channels = full_image.bands
		if n_channels > 1:
			full_image = full_image[0] #1 channel only 
		
		#Generate thumbnail 1					
		ratio = target_W/img_w
		image_thumbnail = full_image.resize(ratio, kernel=pyvips.Kernel.NEAREST)

		#Descriptive statistics 1
		out = pyvips.Image.stats(image_thumbnail)
		out1 = out.numpy()
		statistic_vals = out1[0, :] #stats for all bands together  

		#Downsample (performance boost)
		if scale_factor != 1:
			full_image = full_image.resize(scale_factor) #pyvips.Kernel.LANCZOS3 default
		img_w2 = full_image.width
		img_h2 = full_image.height

		#Flip upside down (optional)
		if flip_ud == 1:
			full_image = full_image.flipver()				
		
		item = [] #stats table row
		item_th = [] #th table row
		item.extend(statistic_vals) #original (always provided)  

		#Transforming input		
		for type in type_list:
			
			if type == "original":
				continue

			else:
				#Finding percentiles, capping and stretching image histogram bottom/top

				thumbnail_positive = image_thumbnail[0] - statistic_vals[0]
				channel_positive = full_image - statistic_vals[0]	

				if type == "log":  #natural log() 					
															
					thumbnail_log = (1 + thumbnail_positive).log()
					channel_positive3 = (1 + channel_positive).log() 					
				
					th_low_input, th_high_input = find_P_thresholds(thumbnail_log, pctOut, 16)		
					channel_positive4 = channel_rescaled(channel_positive3, 0, 1, th_low_input, th_high_input) #normalization	                     
					image_med = channel_positive4.median(filterSize) #median filter   

					log_list.append(image_med) #for stack

				elif type == "linear":
					
					th_low_input, th_high_input = find_P_thresholds(thumbnail_positive, pctOut, 16)		
					channel_positive2 = channel_rescaled(channel_positive, 0, 1, th_low_input, th_high_input)	#normalise                     
					image_med = channel_positive2.median(filterSize) #median filter   

					linear_list.append(image_med) #for stack                                			
							
			#Descriptive statistics 2  
			array1 = pyvips.Image.stats(image_med) #rescaled
			array2 = array1.numpy()
			array3 = array2[0, :] #stats for all bands together                                         
			th_array = [th_low_input, th_high_input] #histogram thresholds

			item.extend(array3) #linear/log statistics         
			item_th.extend(th_array)  

		item1 = item + item_th #add histogram thresholds            
		stats_list.append(item1)           

		#Build info table
		stats_list2 = np.array(stats_list)
		stats_list3 = pd.DataFrame(stats_list2)
		stats_list3.columns = descriptive_cols	
	
	#Save pyramids (longer processing for float)
	for type in type_list:	
		#as tiles
		if type == "original": 
			save_stack(fileList0, tileSize, destDir)
		
		#as montage
		elif type == "linear":			 
			save_transformed_stack_bigtiff(linear_list, type, tileSize, destDir)      
		elif type == "log":
			save_transformed_stack_bigtiff(log_list, type, tileSize, destDir)          

	metadata = generate_tile_config(type_list, img_w2, img_h2, tileSize, destDir) 		

	if save_recoloured == 1:
		#save recoloured chemical maps (montages)		
		destDir2 = os.path.join(destDir, 'recoloured_pctOut' + str(pctOut))        
		mkdir2(destDir2)

		recoloured_list = save_recoloured_stack(fileList0, type_list, linear_list, log_list,
									lut, destDir2, max_workers=4)
	else:
		recoloured_list = []
		print('Recoloured maps skipped.')	

	#Save info table
	file_name1 = os.path.join(destDir, "descriptiveStats.csv")
	descriptiveStats = pd.concat([chosen_table, stats_list3], axis=1)    
	descriptiveStats.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)

	return metadata, descriptiveStats, recoloured_list


def build_pyramids_prior(chosen_table, scale, flip_ud, type_list, workingDir_prior, filterSize, 
						 save_recoloured, lut, tileSize, destDir):
	#This function reproduce RGB colour schema from a previous transformation despite potentially 
	#encountering new outlier values. It uses prior statistics from images used to calculate 
	#a previous model (PCA, DSA, UMAP)
	
	#Default
	target_W = 5000 #same as 'img_rescaled'	
	scale_factor = 1/scale	  
	descriptive_cols = descriptive_colnames(type_list) #pyvips stats table  		

	#Load montages
	fileList0 = chosen_table["path"]
	
	#prior
	statsFile = os.path.join(workingDir_prior, "descriptiveStats.csv")
	descriptiveStats = pd.read_csv(statsFile)    

	log_list = [] #for images
	linear_list = []        
	stats_list = [] #for csv
	for i, filename in enumerate(fileList0):		
		
		basename_ext = os.path.basename(filename)		                                 
		print(f"Reading {basename_ext}")  
		
		descriptiveRow = descriptiveStats.iloc[i, :] #prior		                      
		
		#Load image
		full_image = pyvips.Image.new_from_file(filename) 
		img_w = full_image.width			
		n_channels = full_image.bands
		if n_channels > 1:
			full_image = full_image[0] #1 channel only    

		#Generate thumbnail 1					
		ratio = target_W/img_w
		image_thumbnail = full_image.resize(ratio, kernel=pyvips.Kernel.NEAREST)

		#Descriptive statistics 1
		out = pyvips.Image.stats(image_thumbnail)
		out1 = out.numpy()
		statistic_vals = out1[0, :] #stats for all bands together 

		#Downsample (performance boost)
		if scale_factor != 1:
			full_image = full_image.resize(scale_factor) #pyvips.Kernel.LANCZOS3 default
		img_w2 = full_image.width
		img_h2 = full_image.height

		#Flip upside down (optional)
		if flip_ud == 1:
			full_image = full_image.flipver()	
		
		item = [] #stats table row
		item_th = [] #th table row
		item.extend(statistic_vals) #original (always provided)  

		#Transforming input		
		for type in type_list:					

			if type == "original":
				continue 
			
			else:
				#Capping and stretching image histogram bottom/top

				#Use prior limits
				col1 = f"original_min"
				col2 = f"{type}_th_low_input"
				col3 = f"{type}_th_high_input"
				scaler_input = descriptiveRow[[col1, col2, col3]].to_numpy()	 
				min_temp = scaler_input[0]
				th_low_input = scaler_input[1]
				th_high_input = scaler_input[2]	
				
				image_difference = full_image - min_temp #might be very negative

				if type == "log":  #natural log() 				
					
					#medicine				
					image_positive = 1 + (image_difference < 0).ifthenelse(0, image_difference)					
					channel_positive = image_positive.log() 					

					channel_positive2 = channel_rescaled(channel_positive, 0, 1, th_low_input, th_high_input)	#normalise                     
					image_med = channel_positive2.median(filterSize) #median filter   

					log_list.append(image_med) #for stack

				elif type == "linear":								
					
					channel_positive2 = channel_rescaled(image_difference, 0, 1, th_low_input, th_high_input)	#normalise                     
					image_med = channel_positive2.median(filterSize) #median filter   

					linear_list.append(image_med) #for stack			                                                                                                  

			#Descriptive statistics 2  
			array1 = pyvips.Image.stats(image_med) #rescaled
			array2 = array1.numpy()
			array3 = array2[0, :] #stats for all bands together                                         
			th_array = [th_low_input, th_high_input]

			item.extend(array3) #linear/log statistics         
			item_th.extend(th_array)            			             

		item1 = item + item_th #add histogram thresholds            
		stats_list.append(item1)                      

		#Build info table
		stats_list2 = np.array(stats_list)                
		stats_list3 = pd.DataFrame(stats_list2)            
		stats_list3.columns = descriptive_cols    

	
	#save tiles (longer processing for float)
	for type in type_list:
		if type == "original":
			save_stack(fileList0, tileSize, destDir)
		elif type == "linear":                  
			save_transformed_stack_bigtiff(linear_list, type, tileSize, destDir)      
		elif type == "log":                  
			save_transformed_stack_bigtiff(log_list, type, tileSize, destDir)      

	metadata = generate_tile_config(type_list, img_w2, img_h2, tileSize, destDir) 
	
	#save montages
	if save_recoloured == 1:
		destDir2 = os.path.join(destDir, 'recoloured_prior')        
		mkdir2(destDir2)	

		recoloured_list = save_recoloured_stack(fileList0, type_list, linear_list, log_list,
										  lut, destDir2, max_workers=4)
	else:
		recoloured_list = []
		print('Recoloured maps skipped.')

	#Save info table
	file_name1 = os.path.join(destDir, "descriptiveStats.csv")
	descriptiveStats = pd.concat([chosen_table, stats_list3], axis=1)    
	descriptiveStats.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)

	return metadata, descriptiveStats, recoloured_list

#endregion