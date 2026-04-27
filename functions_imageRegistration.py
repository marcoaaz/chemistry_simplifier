
import os
import pandas as pd
import numpy as np
import pyvips
from helperFunctions.mkdir_options import mkdir2	

def read_img_size(ref_path):
	#Load image    
	ref = pyvips.Image.new_from_file(ref_path) #, access='sequential'
	t_w, t_h = ref.width, ref.height 

	return [t_w, t_h]

def load_bigwarp_landmarks(csv_path):
	# Based on landmarks.csv: 
	
	# Col 0: Point Name ("Pt-0")
	# Col 1: IsActive flag ("true"/"false") 
	# Col 2, 3: Moving (Source) X, Y 
	# Col 4, 5: Fixed (Target) X, Y 

	df = pd.read_csv(csv_path, header=None)
		
	# Filter: Convert column 1 to string, strip any spaces, and check for 'TRUE'
	active_mask = df[1].astype(str).str.strip().str.upper() == 'TRUE'
	df_filtered = df[active_mask]
	
	# Extract the coordinate values
	src_pts = df_filtered[[2, 3]].values.astype(np.float64)
	dst_pts = df_filtered[[4, 5]].values.astype(np.float64)    
	
	return src_pts, dst_pts

def get_exact_affine_matrix(src_pts, dst_pts):
	"""
	Matches BigWarp's 'Affine' behavior using 64-bit Least Squares.
	Returns a 2x3 matrix [[a, b, tx], [c, d, ty]]
	"""
	n = dst_pts.shape[0]
	# Add a column of ones to account for translation (offset)
	design_matrix = np.hstack([dst_pts, np.ones((n, 1))])
	
	# Solve A * M = src_pts 
	sol, _, _, _ = np.linalg.lstsq(design_matrix, src_pts, rcond=None)

	return sol.T

def tps_math(p, src, dst):
	'''
	#Thin plate spline
	#Note: avoids OpenCV precision issues in high-resolution images. No regularisation needed.

	# p: target grid points, 
	# src: src landmarks, 
	# dst: dst landmarks
	'''
	n = src.shape[0]
	
	# K matrix
	diff = dst[:, None, :] - dst[None, :, :]
	r = np.sqrt(np.sum(diff**2, axis=-1))
	K = r**2 * np.log(r + 1e-6)
	# L matrix
	P = np.hstack([np.ones((n, 1)), dst])
	L = np.block([[K, P], [P.T, np.zeros((3, 3))]])
	# Y matrix
	Y = np.vstack([src, np.zeros((3, 2))])
	
	W = np.linalg.solve(L, Y)
	
	# Apply to grid p
	m = p.shape[0]
	diff_p = p[:, None, :] - dst[None, :, :]
	r_p = np.sqrt(np.sum(diff_p**2, axis=-1))
	K_p = r_p**2 * np.log(r_p + 1e-6)
	P_p = np.hstack([np.ones((m, 1)), p])
	
	return K_p @ W[:-3] + P_p @ W[-3:]

def register_wsi(path_moving, size_fixed, csv_path, output_path, 
				 method="affine", interpolation_method="bilinear"):     

	src_pts, dst_pts = load_bigwarp_landmarks(csv_path)
	t_w, t_h = size_fixed[0], size_fixed[1]
	
	#Load
	full_image = pyvips.Image.new_from_file(path_moving, n=-1)
	original_format = full_image.format 
	
	#Medicine
	n_pages = full_image.get("n-pages") if "n-pages" in full_image.get_fields() else 1
	page_height = full_image.get("page-height") if "page-height" in full_image.get_fields() else full_image.height
	
	if n_pages == 3: #weird ImageJ RGB Multi-page structure        
		
		red = full_image.extract_area(0, 0, full_image.width, page_height)[0]
		green = full_image.extract_area(0, page_height, full_image.width, page_height)[0]
		blue = full_image.extract_area(0, page_height * 2, full_image.width, page_height)[0]
		moving = red.bandjoin([green, blue])
	
	else: #typical image
		
		moving = pyvips.Image.new_from_file(path_moving)
		if moving.bands in [2, 4]:
			moving = moving[:moving.bands - 1]
	
	#Math processing
	moving = moving.copy(interpretation="multiband") #requirement
	
	if method in ["similarity", "affine"]:
		model = get_exact_affine_matrix(src_pts, dst_pts)
		a, b, tx = model[0]
		c, d, ty = model[1]        
		coords = pyvips.Image.xyz(t_w, t_h)
		warp_map = (coords[0] * a + coords[1] * b + tx).bandjoin(
					coords[0] * c + coords[1] * d + ty)
		
	elif method == "tps":        
		grid_res = 1000 
		gx, gy = np.meshgrid(np.linspace(0, t_w, grid_res), np.linspace(0, t_h, grid_res))
		grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
		warped_grid_flat = tps_math(grid_pts, src_pts, dst_pts)
		warped_grid = warped_grid_flat.reshape(grid_res, grid_res, 2).astype(np.float32)
		
		raw_map = pyvips.Image.new_from_array(warped_grid).copy_memory()
		raw_map = raw_map.copy(interpretation="multiband")
		
		h_scale = t_w / grid_res
		v_scale = t_h / grid_res        
		warp_map = raw_map.resize(h_scale, vscale=v_scale, kernel="linear")

	warp_map = warp_map.cast("float").copy(interpretation="multiband") 

	#Warping
	warped_bands = []
	for i in range(moving.bands):
		res = moving[i].mapim(warp_map, interpolate=pyvips.Interpolate.new(interpolation_method))
		warped_bands.append(res[0]) 

	# Join bands and cast back to the original bit-depth
	registered = warped_bands[0].bandjoin(warped_bands[1:]) if len(warped_bands) > 1 else warped_bands[0]
	registered = registered.cast(original_format)
	
	#TIF interpretation
	if registered.bands == 3:
		# Setting to srgb forces PhotometricInterpretation = RGB (2)
		registered = registered.copy(interpretation="srgb")
	else:
		# Setting to b-w forces PhotometricInterpretation = MinIsBlack (1)
		registered = registered.copy(interpretation="b-w")

	# Clean up any leftover metadata that might confuse the saver
	if "n-pages" in registered.get_fields():
		registered.remove("n-pages")

	registered.tiffsave(output_path, compression="lzw", tile=True, bigtiff=True)
	# print(f"Saved {registered.bands}-band {original_format} image to {output_path}")


def registration_run(path_csv, size_fixed, moving_image_list, 
					 recoloured_list, representation_list, chosen_table,  
					 transformation_method, interpolation_method, outputFolder):
	
	active_moving = [t for t in ['recoloured', 'represented', 'originals'] if t in moving_image_list]
	
	for moving_type in active_moving:
			
		#Setup Folders
		new_folder = f'registration_{moving_type}'
		outputFolder2 = os.path.join(outputFolder, new_folder)
		mkdir2(outputFolder2)        
		print(f"Saving into {new_folder}")

		if moving_type == 'recoloured':
			temp_list = recoloured_list
		elif moving_type == 'represented':
			temp_list = representation_list
		elif moving_type == 'originals':
			temp_list = chosen_table["path"]

		for path_moving in temp_list:            
			
			input_basename = os.path.splitext(os.path.basename(path_moving))[0]
			output_path = os.path.join(outputFolder2, f'{input_basename}_{transformation_method}.tif')

			register_wsi(path_moving, size_fixed, path_csv, output_path, 
							transformation_method, interpolation_method)
			
		