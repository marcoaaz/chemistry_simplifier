
import os
import pandas as pd
import numpy as np
import pyvips

from functions_imageRegistration import register_wsi, read_img_size

# path_moving = os.path.join(folder, "moving.jpg")
# path_fixed = os.path.join(folder, "fixed.jpg")
# path_csv = os.path.join(folder, "landmarks2.csv")

# folder = r"C:\Users\acevedoz\OneDrive - Queensland University of Technology\Desktop\test2"
# path_moving = os.path.join(folder, "20BSK-001B - BSE 17-31.5.png") #"20BSK-001B - BSE 17-31.5.png"
# path_fixed = os.path.join(folder, "xpl_max_z0_flat2.tif")
# path_csv = os.path.join(folder, "landmarks_bse_xpl.csv")


# --- Configuration ---

path_moving = r"C:\Users\acevedoz\OneDrive - Queensland University of Technology\Desktop\outputs_advanced samples\1_chemistry-simplifier_outputs\MS_MLY053C - GSQ #1_BSE_BSE-1.tif"
path_fixed = r"C:\Users\acevedoz\OneDrive - Queensland University of Technology\Desktop\outputs_advanced samples\Leyshon_MS_MLY053c\10x_RL BF_01 #6_z0.tif"
path_csv = r"C:\Users\acevedoz\OneDrive - Queensland University of Technology\Desktop\outputs_advanced samples\1_chemistry-simplifier_outputs\MS_MLY053C_pca_8bit_landmarks.csv"

transform_type = "tps"  #tps, affine, similarity
interpolation_method = "bilinear" #nearest, bilinear (fast), lbb (spline), bicubic (slow)
output_file = os.path.join(os.path.dirname(path_fixed), f'registered_{transform_type}.tif')

size_fixed = read_img_size(path_fixed)			
t_w = size_fixed[0]
t_h = size_fixed[1]
size_fixed = [t_w, t_h]

register_wsi(path_moving, size_fixed, path_csv, output_file, transform_type, interpolation_method)

# moving = pyvips.Image.new_from_file(path_moving)
# print(f"--- Diagnostic for: {os.path.basename(path_moving)} ---")
# print(f"Total Bands: {moving.bands}")
# print(f"Interpretation: {moving.interpretation}")

# for i in range(moving.bands):
#     band = moving[i]
#     print(f"Band {i} -> Max: {band.max()}, Avg: {band.avg()}")