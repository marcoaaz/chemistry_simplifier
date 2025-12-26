# -*- coding: utf-8 -*-
"""
functions_pyramids.py

Script useful for tiling and stacking chemical layers of very large (WSI) X-ray maps.

Citation: This is the update of the first half of the first script for dimensionality reduction 
'stackingCPS_synchrotron_v6.m’ in Acevedo Zamora et al. (2024). https://doi.org/10.1016/j.chemgeo.2024.121997

Current version: python 3.13.7, chemSimplifier2
Written by: Marco Andres, ACEVEDO ZAMORA

Updates:
Published (first version): 14-Nov-23
second version for whole-slide imaging: 19-Jun-24, 5-Sep-24
third version for GUI: 16-sep-25, 30-Sep-25, 9-Oct-25

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
from PIL import Image #to remove pyvips metadata

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

## Functions

#region Helper functions

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

	#Edit to be similar
	fileList = glob.glob(f"{imageFolder}/*.{extension}") #perfect match needed with 'pattern'    
	pattern = re.compile(r".+\\" + expression + extension2)        
	
	#print(np.transpose(np.array(fileList)))           

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

def predict_fileSize(dim, fileSize_th):
	#fraction of WSI for training (prevents RAM overload) 
	#pytorch issue:  
	#OverflowError: cannot serialize a bytes object larger than 4 GB          

	imageHeight = dim[0]
	imageWidth = dim[1]
	n_layers_input = dim[2]

	#Predict file size
	bitDepth = (2**8)
	fileSize = (10**(-9))*(imageHeight*imageWidth*n_layers_input*bitDepth/8) #GB    

	if fileSize > fileSize_th:
		fraction = fileSize_th/fileSize    
	else:
		fraction = 1

	# print(f"mosaic of WxHxC= {imageWidth}x{imageHeight}x{n_layers_input} px")     

	return fraction
	
#endregion

#region Image processing 

def clear_all_metadata(imgname): 	
	#The output will not have an ImageJ pixel calibration that conflicts with 
	# control point registration (Fiji > BigWarp) referenced to the 'moving' image control points
	#by Muhammad Abdullahi: 
	#https://thepythoncode.com/article/how-to-clear-image-metadata-in-python

    img = Image.open(imgname)
    
	# Read the image data, excluding metadata.
    data = list(img.getdata())
    
	# Create a new image with the same mode and size but without metadata.
    img_without_metadata = Image.new(img.mode, img.size)
    img_without_metadata.putdata(data)
    
	# Overwrite. 
    img_without_metadata.save(imgname)

	
def save_recoloured_channel(image_med, lut, destFile1):

	image_recoloured = image_med.maplut(lut)
	image_recoloured.write_to_file(destFile1)

#Extract from ray_tracing_module.py 
def channel_uint8(image_rs3):
	image_rs4 = image_rs3.cast("uchar") #uint8    

	return image_rs4

def find_P_thresholds(input_channel, percentOut, bit_precision):
	#bit_precision: 8-bit=255; 16-bit=65535
	
	percentOut1 = percentOut + 0.005 #medicine
	# percentOut=0 Error: "65536" of type 'gint' is invalid or out of range for property 'threshold' of type 'gint'

	calc_depth = (2**bit_precision) -1     
	min_val = input_channel.min()
	max_val = input_channel.max()
	 
	#medicine 1: zero division if weight-decay is too low and no. epochs too high
	range_val = max((max_val - min_val), 0.1) 
	ratio = (range_val/calc_depth)

	#Finding percentiles	
	image_rs1 = (input_channel - min_val) / ratio #0-65355         
	image_rs2 = image_rs1.cast("uint") #'uchar'=8-bit, 'uint' or 'ushort'=16-bit       
		
	th_low = image_rs2.percent(percentOut1) #'int'	
	th_high = image_rs2.percent(100 - percentOut1)        
	
	#values	
	th_low_input = th_low*ratio + min_val 
	th_high_input = th_high*ratio + min_val                    

	#medicine 2: zero division if you are processing an artefact image (e.g., Synchrotron data Flux0)
	if th_low_input == th_high_input:
		th_high_input = th_high_input + 1

	return th_low_input, th_high_input

def channel_rescaled(input_channel, min_val, max_val, th_low_input, th_high_input):  
	#Following: https://au.mathworks.com/help/matlab/ref/rescale.html
	#Note: capped and uint8 (useful for std, maxIndex, minIndex, PCA)	 
	  
	# min_val = 0 #default
	# max_val = 255

	#Capping
	input_channel = (input_channel > th_high_input).ifthenelse(th_high_input, input_channel) #true, false
	input_channel = (input_channel < th_low_input).ifthenelse(th_low_input, input_channel)

	#Rescaling
	output_channel = min_val + (input_channel - th_low_input) * ( (max_val - min_val) / (th_high_input - th_low_input) ) 			
	# output_channel = image_rs3.cast("uchar") #uint8   	

	return output_channel   

def channel_standardise(input_channel, mean_temp, std_temp):
	
	output_channel = (input_channel - mean_temp) / std_temp #z-score		

	return output_channel  

#Similar to vsiFormatter > ray_tracing_module.py
def img_rescaled(image_cropped, percentOut):	
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
	file_output2 = os.path.join(outputFolder, "descriptiveStats_" + str2 + ".csv")	
	descriptiveStats.to_csv(file_output2, sep=',', encoding='utf-8', index=False, header=True)


def stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder):    

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
	path1 = os.path.dirname(modelPATH)
	statsFile = os.path.join(path1, "descriptiveStats_" + str2 + ".csv")#chosen_table must match
	descriptiveRow = pd.read_csv(statsFile) #input stats
	
	image_rescaled, descriptiveStats = img_rescaled_prior(image_cropped, descriptiveRow) #float
	image_rescaled_int = image_rescaled.cast("uchar") #uint8  
	image_rescaled_int2 = image_rescaled_int.copy()	#xres (inmutable)

	#Output montage	
	file_output = os.path.join(outputFolder, 'montage_' + str2 + '.tif')
	file_output2 = os.path.join(outputFolder, 'montage_' + str2 + '_8bit.tif')

	image_rescaled.write_to_file(file_output) 
	image_rescaled_int2.write_to_file(file_output2)	#resunit= cm, inch (mandatory)	
	clear_all_metadata(file_output2) #medicine

	#Output stats (for reproducibility)	
	file_output2 = os.path.join(outputFolder, "descriptiveStats_" + str2 + "_prior.csv")	
	descriptiveStats.to_csv(file_output2, sep=',', encoding='utf-8', index=False, header=True)


#endregion

#region Save Deep Zoom tiles

def learn_tileConfiguration(destDir):
	# scan tileset    
	fileList = glob.glob(f"{destDir}/*/*/*_*.tif")
	pattern = re.compile(r".*\\(\w+)_pyramid_files\\0\\(\d+)_(\d+)\.tif") 
	#r".*/(\d+)_(\d+)\.tif" in Linux
	
	out2 = []
	for filename in fileList:
		match = pattern.match(filename)

		if match:     
			#Parsed info
			type = match.group(1)
			x = match.group(2)
			y = match.group(3)

			#dim
			image_temp = pyvips.Image.new_from_file(filename)
			W = image_temp.width
			H = image_temp.height

			out2.append([filename, type, x, y, W, H])

	out3 = np.array(out2) 
	file_table = pd.DataFrame(out3)
	file_table.columns =['filepath', 'type', 'x', 'y', 'W', 'H']    
	#medicine
	file_table['x'] = file_table['x'].astype(int)
	file_table['y'] = file_table['y'].astype(int)
	file_table2 = file_table.sort_values(['type', 'y', 'x'], ascending=[True, True, True]) #for pyvips.Image.arrayjoin

	file_name1 = os.path.join(destDir, "tileConfiguration.csv")
	file_table2.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)

	#Console
	tiles_down = int(file_table2['y'].max()) + 1
	tiles_across = int(file_table2['x'].max()) + 1
	print(f"Deep Zoom pyramid with {tiles_down}x{tiles_across} tiles")

	return file_table2

def save_stack(path_list, tileSize, flip_ud, destDir):
	#Saving original stack (used for 'ROIimageAnalysis_v7_wsi.m' script)

	destDir4 = os.path.join(destDir, 'original_pyramid')
	destDir4_files = destDir4 + '_files'
	try:
		remove(destDir4_files) #and subfolders
	except:
		print("Producing original pyramid for the first time")

	#Loading tiles
	pages = [pyvips.Image.new_from_file(path) for path in path_list]
	image_stack = pages[0].bandjoin(pages[1:])

	#Medicine (fixing pyvips vertical flip)
	if flip_ud == 1:
		image_flipped = image_stack.flipver()
	elif flip_ud == 0:
		image_flipped = image_stack    

	image_flipped.dzsave(destDir4, suffix='.tif', 
					skip_blanks=-1, background=0, 
					depth='one', overlap=0, tile_size= tileSize, 
					layout='dz') #Tile overlap in pixels*2

def save_transformed_stack(stack_layers_log, type, tileSize, flip_ud, destDir):
	#Saving linear/log transformed stack (uint8) 
	#Output used of PCA analysis 'wsi_dimPCA_v1.m' and autoencoder 'DSA_wsi_training_v3.py'
	
	if type == 'log': #natural log
		destination_folder = 'log_pyramid'
	elif type == 'linear':
		destination_folder = 'linear_pyramid'    

	destDir3 = os.path.join(destDir, destination_folder)
	destDir3_files = destDir3 + '_files'
	try:
		remove(destDir3_files)
	except:
		print(f"Saving {type} tiles for the first time")

	#Transformed data
	image_stack_log_recoloured = stack_layers_log[0].bandjoin(stack_layers_log[1:])    

	#Medicine: fixing pyvips vertical flip on Synchrotron TIFs
	if flip_ud == 1:
		image_log_flipped = image_stack_log_recoloured.flipver()
	elif flip_ud == 0:
		image_log_flipped = image_stack_log_recoloured
	
	
	image_log_flipped.dzsave(destDir3, suffix='.tif', 
					skip_blanks=-1, background=0, 
					depth='one', overlap=0, tile_size= tileSize, 
					layout='dz') #Tile overlap in pixels*2  
	
def build_pyramids(chosen_table, type_list, tileSize, filterSize, pctOut, destDir, save_recoloured, flip_ud):
		
	fileList0 = chosen_table["path"]

	#Preparing false-colour images
	destDir2 = os.path.join(destDir, 'recoloured'+ '_pctOut' + str(pctOut))        
	mkdir2(destDir2)

	#Define colour map
	img_indexes = pyvips.Image.identity()
	lut = img_indexes.falsecolour() #using standard heatmap
	#256x1 uchar, 3 bands, srgb, pngload
	
	#Preparing pyvips metadata table    
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

	#Load montages
	log_list = [] #for images
	linear_list = []        
	stats_list = [] #for csv
	for filename in fileList0:

		basename_ext = os.path.basename(filename)
		basename = os.path.splitext(basename_ext)[0]                                    
		print(f"Processing {basename_ext}")                        
		
		#Load image
		image = pyvips.Image.new_from_file(filename) 
		
		#Ensure layers have 1 channel
		n_channels = image.bands
		if n_channels > 1:
			image = image[0]     

		#Descriptive statistics 1
		out = pyvips.Image.stats(image)
		out1 = out.numpy()
		statistic_vals = out1[0, :]
		
		item = [] #stats table row
		item_th = [] #th table row
		item.extend(statistic_vals) #original (always provided)  

		#Transforming input		
		for type in type_list:

			if type == "log":  #natural log() 
				image_positive = 1 + image - statistic_vals[0]
				image_temp = image_positive.log() 
				
				#Finding percentiles, capping and stretching image histogram bottom/top                
				th_low_input, th_high_input = find_P_thresholds(image_temp, pctOut, 16)		
				channel_positive2 = channel_rescaled(image_temp, 0, 1, th_low_input, th_high_input)	                     
				image_med = channel_positive2.median(filterSize) #median filter   

				log_list.append(image_med) #for stack

			elif type == "linear":
				image_temp = image - statistic_vals[0]			

				#Finding percentiles, capping and stretching image histogram bottom/top
				th_low_input, th_high_input = find_P_thresholds(image_temp, pctOut, 16)		
				channel_positive2 = channel_rescaled(image_temp, 0, 1, th_low_input, th_high_input)	                     
				image_med = channel_positive2.median(filterSize) #median filter   

				linear_list.append(image_med) #for stack

			elif type == "original":
				continue                                                                                                 

			#Descriptive statistics 2  
			array1 = pyvips.Image.stats(image_med) #rescaled
			array2 = array1.numpy()
			array3 = array2[0, :] #stats for all bands together                                         
			item.extend(array3) #linear/log statistics         

			#Histogram thresholds  
			th_array = [th_low_input, th_high_input]
			item_th.extend(th_array)                 

			#Saving recoloured images (for retrospective feedback): time-consuming                 
			if save_recoloured == 1:                        
				image_med2 = (255*image_med).cast("uchar") #uint8                      
				
				destFile1 = os.path.join(destDir2, f"{basename}_{type}.tif")                
				save_recoloured_channel(image_med2, lut, destFile1)                 

		item1 = item + item_th #add histogram thresholds            
		stats_list.append(item1)                     

		#Build info table
		stats_list2 = np.array(stats_list)                
		stats_list3 = pd.DataFrame(stats_list2)            
		stats_list3.columns = descriptive_cols                       

	#Save info table
	file_name1 = os.path.join(destDir, "descriptiveStats.csv")
	descriptiveStats = pd.concat([chosen_table, stats_list3], axis=1)    
	descriptiveStats.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)
	
	#save tiles (longer processing for float)
	for type in type_list:
		if type == "original":
			save_stack(fileList0, tileSize, flip_ud, destDir)
		elif type == "linear":                  
			save_transformed_stack(linear_list, type, tileSize, flip_ud, destDir)      
		elif type == "log":                  
			save_transformed_stack(log_list, type, tileSize, flip_ud, destDir)          

	metadata = learn_tileConfiguration(destDir) #after saving (required)

	return metadata, descriptiveStats

def build_pyramids_prior(chosen_table, type_list, tileSize, filterSize, workingDir_prior, destDir, save_recoloured, flip_ud):
	#Note: unlike build_pyramids(), this function uses the statistics from prior images used to calculate previous model
	#This allows to fully reproduce the RGB colour schema from the colour transformation (PCA, DSA, UMAP) despite the presence of outlier values

	fileList0 = chosen_table["path"]

	#prior
	statsFile = os.path.join(workingDir_prior, "descriptiveStats.csv")
	descriptiveStats = pd.read_csv(statsFile)    	

	#Preparing false-colour images
	destDir2 = os.path.join(destDir, 'recoloured'+ '_prior')        
	mkdir2(destDir2)

	#Define colour map
	img_indexes = pyvips.Image.identity()
	lut = img_indexes.falsecolour() #using standard heatmap
	#256x1 uchar, 3 bands, srgb, pngload      

	#Preparing pyvips metadata table    
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

	#Load montages as large stack
	log_list = []
	linear_list = []        
	stats_list = [] #for csv
	for i, filename in enumerate(fileList0):		
		
		descriptiveRow = descriptiveStats.iloc[i, :] #prior

		basename_ext = os.path.basename(filename)
		basename = os.path.splitext(basename_ext)[0]                                    
		print(f"Processing {basename_ext}")                        
		
		#Load image
		image = pyvips.Image.new_from_file(filename) 
		
		#Ensure layers have 1 channel
		n_channels = image.bands
		if n_channels > 1:
			image = image[0]     

		#Descriptive statistics 1
		out = pyvips.Image.stats(image)
		out1 = out.numpy()
		statistic_vals = out1[0, :]
		
		item = [] #stats table row
		item_th = [] #th table row
		item.extend(statistic_vals) #original (always provided)  

		#Transforming input		
		for type in type_list:

			#Use prior limits
			col1 = f"original_min"
			col2 = f"{type}_th_low_input"
			col3 = f"{type}_th_high_input"
			scaler_input = descriptiveRow[[col1, col2, col3]].to_numpy()	 
			min_temp = scaler_input[0]
			th_low_input = scaler_input[1]
			th_high_input = scaler_input[2]

			if type == "log":  #natural log() 
				image_positive = 1 + image - min_temp
				image_temp = image_positive.log() 
				
				#Capping and stretching image histogram bottom/top                                
				channel_positive2 = channel_rescaled(image_temp, 0, 1, th_low_input, th_high_input)	                     
				image_med = channel_positive2.median(filterSize) #median filter   

				log_list.append(image_med) #for stack

			elif type == "linear":
				image_temp = image - min_temp				

				#Capping and stretching image histogram bottom/top                
				channel_positive2 = channel_rescaled(image_temp, 0, 1, th_low_input, th_high_input)	                     
				image_med = channel_positive2.median(filterSize) #median filter   

				linear_list.append(image_med) #for stack

			elif type == "original":
				continue                                                                                                 

			#Descriptive statistics 2  
			array1 = pyvips.Image.stats(image_med) #rescaled
			array2 = array1.numpy()
			array3 = array2[0, :] #stats for all bands together                                         
			item.extend(array3) #linear/log statistics         

			#Histogram thresholds  
			th_array = [th_low_input, th_high_input]
			item_th.extend(th_array)            

			#Saving recoloured images (for retrospective feedback): time-consuming                 
			if save_recoloured == 1:                        
				image_med2 = (255*image_med).cast("uchar") #uint8                      
				
				destFile1 = os.path.join(destDir2, f"{basename}_{type}.tif")                
				save_recoloured_channel(image_med2, lut, destFile1)                 

		item1 = item + item_th #add histogram thresholds            
		stats_list.append(item1)                      

		#Build info table
		stats_list2 = np.array(stats_list)                
		stats_list3 = pd.DataFrame(stats_list2)            
		stats_list3.columns = descriptive_cols                      

	#Save table
	file_name1 = os.path.join(destDir, "descriptiveStats.csv")
	descriptiveStats = pd.concat([chosen_table, stats_list3], axis=1)    
	descriptiveStats.to_csv(file_name1, sep=',', encoding='utf-8', index=False, header=True)
	
	#save tiles (longer processing for float)
	for type in type_list:
		if type == "original":
			save_stack(fileList0, tileSize, flip_ud, destDir)
		elif type == "linear":                  
			save_transformed_stack(linear_list, type, tileSize, flip_ud, destDir)      
		elif type == "log":                  
			save_transformed_stack(log_list, type, tileSize, flip_ud, destDir)      

	metadata = learn_tileConfiguration(destDir) #after saving (required)
	
	return metadata, descriptiveStats

#endregion