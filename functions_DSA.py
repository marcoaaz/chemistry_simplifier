'''
functions_DSA.py

Contains the functions for training and running the Deep Sparse Autoencoder model within Chemistry Simplifier v1 executable.

Documentation:
https://debuggercafe.com/sparse-autoencoders-using-kl-divergence-with-pytorch/
https://docs.pytorch.org/vision/0.22/generated/torchvision.transforms.ToTensor.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
https://docs.pytorch.org/docs/stable/generated/torch.linalg.norm.html
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

Created: 19-Sep-2025, Marco Acevedo
Updated: 29-Sep-25, 9-Oct-25

Notes: ensure model and data are both in the same device (CPU or GPU) for performance

'''

#Dependencies
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')

from multiprocessing import Pool

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

#region Input definition

def incremental_loading_DSA(fileList, scale, fraction, test_ratio):

	#Default    
	scale_factor = 1/scale
	seed_value = 0 #seed for subsetting arrays in training/test sets            
	np.random.seed(seed_value)   
	# torch.manual_seed(seed_value) #for reproducible training         	

	#Load random sample of tile pixels for DSA	
	tile_arrays = []
	for file in fileList:
   
		image_temp = pyvips.Image.new_from_file(file) #uint8 tiles

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
					
		image_xyz2 = image_xyz[sampling_idx, :] #selected data	
		
		tile_arrays.append(image_xyz2)    
	
	#Sub-sample
	data4 = np.concatenate([i for i in tile_arrays])
	n_rows_sel = data4.shape[0]        
	
	#Dataset splitting
	to_row = int(n_rows_sel*(1 - test_ratio))
	indices = np.random.permutation(n_rows_sel)
	training_idx, test_idx = indices[:to_row], indices[to_row:]
	training_x, test_x = data4[training_idx,:], data4[test_idx,:]
	training_y, test_y = data4[training_idx, :], data4[test_idx, :]

	dataset_list = [training_x, training_y, test_x, test_y]

	return dataset_list

class MyDataset(Dataset): #convert numpy to tensor
	def __init__(self, data, targets, transform=None):
		self.data = data		
		self.targets = targets
		self.transform = transform
		
	def __getitem__(self, index): #index probably for DataLoader
		x = self.data[index]
		y = self.targets[index]           
		
		if self.transform:            			
			x = self.data[index] #np array            
			x = self.transform(x) #convert to a torch.FloatTensor 

		return x, y
	
	def __len__(self):
		return len(self.data)
	
# endregion  

#region Autoencoder model

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()		

		self.linear0 = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
		self.af0 = nn.Sigmoid() #non-linear activation function (probability distribution [0-1] for sparsity term)		

	def forward(self, x):
		x = self.linear0(x)
		x = self.af0(x)		
		return x

#Symmetrical architecture
class SparseAutoencoder(nn.Module):
	def __init__(self, network_nodes):

		n_layers_input = network_nodes[0]
		nodes_layer1 = network_nodes[1]
		nodes_layer2 = network_nodes[2]
		n_channels_bottleneck = network_nodes[3]				
		do_probability = [0, 0] #no dropout		
		# do_probability = [0.2, 0.3] #recommended for large NN

		super(SparseAutoencoder, self).__init__() #same order as model.children()  	
		self.do0 = nn.Dropout(p= do_probability[0])
		self.do1 = nn.Dropout(p= do_probability[1])

		#encoder
		self.block1 = BasicBlock(in_channels=n_layers_input, out_channels=nodes_layer1) 
		self.block2 = BasicBlock(in_channels=nodes_layer1, out_channels=nodes_layer2)
		self.block3 = BasicBlock(in_channels=nodes_layer2, out_channels=n_channels_bottleneck) #bottleneck
		#decoder
		self.block4 = BasicBlock(in_channels=n_channels_bottleneck, out_channels=nodes_layer2) 
		self.block5 = BasicBlock(in_channels=nodes_layer2, out_channels=nodes_layer1)
		self.outputLayer = nn.Linear(in_features=nodes_layer1, out_features=n_layers_input, bias=True)        
	
	def forward(self, x):		
		x = self.block1(x)
		x = self.do0(x)
		x = self.block2(x)
		x = self.do1(x)
		x = self.block3(x)
		x = self.do1(x)

		x = self.block4(x)
		x = self.do1(x)
		x = self.block5(x)
		x = self.do1(x)
		x = self.outputLayer(x)

		return x

#Cost regularization penalty strength functions    

#Kullback-Leibler for node activation sparsity term
def kl_divergence(sparsity_target, rho_hat0, device):	
	#rho_hat0 = activations in each layer for entire batch		
	
	#chosen running average
	# rho_hat = torch.mean(rho_hat0, 0) #for hidden node accumulation
	rho_hat = torch.mean(rho_hat0, 1) #for hidden layer accumulation			
	rho = torch.full_like(rho_hat, sparsity_target)

	epsilon = 1e-8
	kl_divergence_term = F.kl_div((rho_hat + epsilon).log(), rho + epsilon, 
							   reduction='batchmean' #mathematical definition
							   )

	# #Manual calculation (used in publication)
	# rho = torch.tensor([sparsity_target] * len(rho_hat)).to(device)
	# kl_divergence_term = torch.sum( rho*torch.log(rho/rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat)) )
	
	return kl_divergence_term

def sparse_loss2(rho, values, model_children, device):		
	# (not node/units in the whole training set)
	
	children_list = range(2, 7) #first 2 are dropout layers, the last the output layer
	loss = 0
	for i in children_list:

		values = model_children[i](values)						
		loss += kl_divergence(rho, values, device)

	return loss

#Tikhonov L2 for weight-decay term
def weight_loss1(model):   

	weight_params = []
	bias_params = []
	for name, param in model.named_parameters():
		if 'bias' in name:
			bias_params.append(param)
		else:
			weight_params.append(param)

	l2_norm = torch.tensor(0., requires_grad=True)
	for param1 in weight_params:
		l2_norm = l2_norm + torch.linalg.norm(param1, ord=2) #flattened input, single output
	
	l2 = 0.5*(l2_norm**2)
	
	#Avoid very high learning rate: During the SVD execution, batches 0 failed to converge. 
	#C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cuda\linalg\BatchLinearAlgebraLib.cpp:836.

	return l2

#endregion
 
#region Training function

def fit(model, dataloader, n_trainset, criterion, optimizer, ADD_SPARSITY, BETA, RHO, device):
	
	model.train()
	model_children = list(model.children()) # get the layers 				
	
	running_loss = 0.0    
	n_batches = int(n_trainset/dataloader.batch_size) + 1 #floor    
	for i, data_tuple in tqdm(enumerate(dataloader), total= n_batches, ascii=True):		

		img, _ = data_tuple #tuple (data, target)
		img = img.to(device) #torch.Size([8192, 1, 9, 1])        
		img = img.view(img.size(0), -1) #flattening, torch.Size([8192, 9])        	
		n_pixels = img.size(0)

		#Cost function (tensor)
		outputs = model(img)        
		mse_loss = criterion(outputs, img) #for loss(input, target/labels)		

		if ADD_SPARSITY == 'yes':      
			kl_loss = sparse_loss2(RHO, img, model_children, device)                 
			loss = mse_loss + BETA*kl_loss #dominated by BETA

		elif ADD_SPARSITY == 'no':
			loss = mse_loss  		

		#backpropagation			
		loss.backward()  #compute gradients
		optimizer.step() #update weights with optimiser
		optimizer.zero_grad() #reset the gradients for new batch      

		iteration_loss = loss.item() #get numerical value from tensor		
		running_loss = running_loss + n_pixels*(iteration_loss) 	
	
	epoch_loss = running_loss / n_trainset #weighted average
	print(f"Train Loss: {epoch_loss:.3f}")
			
	return epoch_loss

# Validation function

def validate(model, dataloader, n_testset, criterion, device):	
	model.eval()
	
	running_loss = 0.0
	n_batches = int(n_testset/dataloader.batch_size) + 1 #floor  
	with torch.no_grad():
		for i, data in tqdm(enumerate(dataloader), total=n_batches, ascii=True):
			
			img, _ = data
			img = img.to(device)
			img = img.view(img.size(0), -1)
			n_pixels = img.size(0)
			
			#Cost function
			outputs = model(img)
			mse_loss = criterion(outputs, img)

			loss = mse_loss

			iteration_loss = loss.item() #get numerical value from tensor		
			running_loss = running_loss + n_pixels*(iteration_loss) 
			
	epoch_loss = running_loss / n_testset #weighted average
	print(f"Val Loss: {epoch_loss:.3f}")          
	
	return epoch_loss

#endregion

#region Launch training 

def incremental_training_DSA(dataset_list, BATCH_SIZE, network_nodes, LEARNING_RATE, epoch_default, 
							 ADD_SPARSITY, ALPHA, BETA, RHO, device, n_workers, outputFolder):
	#Default    
	training_x = dataset_list[0]
	training_y = dataset_list[1] 
	test_x = dataset_list[2]
	test_y = dataset_list[3]     
	
	#pytorch dataset (already [0-1]) 	
	loader_transform = v2.Compose([ v2.ToDtype(torch.float32, scale=False), v2.Normalize(mean=(0.5,), std= (0.5,)) ])   

	trainset = MyDataset(training_x, training_y, transform= loader_transform) 
	testset = MyDataset(test_x, test_y, transform= loader_transform)
	
	#forming batches (read as iterator with step)     
	trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers= n_workers)     
	testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers= n_workers)          
	n_trainset = len(trainset)
	n_testset = len(testset)        

	#Generate model
	model = SparseAutoencoder(network_nodes).to(device)    
	
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=ALPHA) 
	# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #old

	criterion = nn.MSELoss() #loss function mean squared error    
	# criterion = nn.CrossEntropyLoss() #preferred for classification
	
	start = time.time()
	train_loss = []
	val_loss = []    	
	for epoch in range(epoch_default):
		print(f"Epoch {epoch+1} of {epoch_default}")        
		
		train_epoch_loss = fit(model, trainloader, n_trainset, criterion, optimizer, ADD_SPARSITY, BETA, RHO, device)		
		val_epoch_loss = validate(model, testloader, n_testset, criterion, device)

		train_loss.append(train_epoch_loss)
		val_loss.append(val_epoch_loss)
	
	end = time.time()
	print(f"DSA training and validation took {(end-start)/60:.3} min")
	
	#Save the model    
	modelPATH = f"{outputFolder}/model_epochs{epoch_default}.tar"
	lossPlotPATH = f"{outputFolder}/loss_plot.png"   

	torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss}, modelPATH)   
	
	save_loss_plot(train_loss, val_loss, lossPlotPATH)

def save_loss_plot(train_loss, val_loss, lossPlotPATH):
	
	#Save loss plot
	plt.figure(figsize=(10, 7))
	plt.plot(train_loss, color='orange', label='training loss')
	plt.plot(val_loss, color='red', label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(lossPlotPATH)
	# plt.show()

# endregion  

#region Predicting

# Prediction functions
def embedded_space(values, model_children):	        	
	
	#Recursive loop targeting bottleneck nodes
	children_list = range(2, 5) #first 2 are dropout layers
	for i in children_list:		
			
		values = model_children[i](values) #issue in GPU: 0.5 activations when weights=0				

	return values 

def predict_space(model, dataloader, device):

	model.eval() #inferencing mode (set dropout and batch normalisation to evaluation mode)	     	
	
	with torch.no_grad(): #prevent gradient computations

		model_children = list(model.children()) # get the layers  		

		pixel_batch = []
		for i, data_tuple in enumerate(dataloader):			

			img, _ = data_tuple
			img2 = img.to(device) #ensuring model is in the same device
			img3 = img2.view(img2.size(0), -1)
			
			output = embedded_space(img3, model_children) 
			
			pixel_batch.append(output)
		
	return pixel_batch 

def _unpack_and_call(args):
	return dsa_section(*args)

def transform_tiles_dsa(fileList, resolution, modelPATH, network_nodes, 
						BATCH_SIZE_pred, device, n_cores, outputFolder2):  	
	print('Predicting tiles')

	#Transform each tile
	total_len = len(fileList) #number of iterations
	args = [(file, resolution, modelPATH, network_nodes, BATCH_SIZE_pred, device, outputFolder2) for file in fileList]		
	
	with Pool(processes= n_cores) as pool: # imap for incremental results        
		fileList2 = list( tqdm(pool.imap(_unpack_and_call, args, chunksize=1), total= total_len, ascii=True) )	

	return fileList2

def dsa_section(file, resolution, modelPATH, network_nodes, BATCH_SIZE_pred, device, outputFolder2):	         
	#Default
	n_channels = 3
	n_workers_pred = 0
	resolution_factor = 1/resolution

	#Initialize    
	checkpoint = torch.load(modelPATH, map_location=device) #dictionary containing objects
	model_pred = SparseAutoencoder(network_nodes).to(device)
	model_pred.load_state_dict(checkpoint['model_state_dict']) #size mismatch (if not adequate model)	

	#Load tile
	image_temp = pyvips.Image.new_from_file(file)
	
	#Downsampling
	image_temp = image_temp.resize(resolution_factor)	

	#To numpy matrix
	image_np = image_temp.numpy()
	dim_shape = image_np.shape  
	image_xyz2 = np.reshape(image_np, (-1, dim_shape[2])) #e.g., 39 channels          	               	

	#pytorch dataset (already [0-1])     
	loader_transform = v2.Compose([ v2.ToDtype(torch.float32, scale=False), v2.Normalize(mean=(0.5,), std= (0.5,)) ])   

	all_set = MyDataset(image_xyz2, image_xyz2, transform= loader_transform)
	alldata_loader = DataLoader(all_set, batch_size= BATCH_SIZE_pred, num_workers= n_workers_pred)   
	n_all_set = len(all_set)

	#DSA transformation
	pixel_batch = predict_space(model_pred, alldata_loader, device) #batch list of bottleneck nodes        
	result = torch.cat(pixel_batch, dim=0)  #torch.Size([n_pixels, 3])   
	
	outputs = result.view(dim_shape[0], dim_shape[1], n_channels).cpu().numpy() #double
	
	image_output = pyvips.Image.new_from_array(outputs) #3-channel image  
	image_output2 = image_output.cast("float") #note: 64-bit floating-point fails reload

	#output file
	output_filename = os.path.basename(file)
	output_path = os.path.join(outputFolder2, output_filename)
	image_output2.write_to_file(output_path) 
	
	return output_path

