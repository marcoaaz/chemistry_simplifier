'''
main.py

Version 1 of software GUI processing multi-elemental chemical maps into Principal Component Analysis (PCA),
Deep Sparse Autoencoder (DSA), and Uniform Manifold Approximation and Projection whole-slide image representations.

Documentation:
https://www.youtube.com/watch?v=2EjrLpC4cE4&t=163s
https://pyinstaller.org/en/stable/usage.html

Written in python 3.9.13
environment: chemSimplifier2

Logo: https://stock.adobe.com/search?k=color+wheel+logo&asset_id=56781933

Created: 19-Sep-25, Marco Acevedo
Updated: 23-Sep-25, 30-Sep-25

Current: python 3.9.13 (chemSimplifier3)

'''

#Essential dependencies
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from chemistrySimplifier_v3 import Ui_MainWindow #relative path

class Window(QMainWindow, Ui_MainWindow):

	#region GUI
	def __init__(self):
		super().__init__()
		self.setupUi(self)

		#Recovering images	
		# relative_path = sys._MEIPASS #PyInstaller executable
		bundle_dir = os.path.abspath(os.path.dirname(__file__)) #relative path
		icon_file_path = os.path.join(bundle_dir, "icons/colour_wheel.ico")
		image_file_path0 = os.path.join(bundle_dir, "icons/QUT-Logo.png")
		image_file_path1 = os.path.join(bundle_dir, "icons/AuScope_logo.png")		

		#Window
		self.setWindowTitle("Chemistry Simplifier v1.0")
		self.setWindowIcon(QtGui.QIcon(icon_file_path))
		self.setMinimumSize(600, 600)
		self.setWindowFlags(self.windowFlags()) 

		#Image update		
		self.label_53.setPixmap(QtGui.QPixmap(image_file_path1))
		self.label_41.setPixmap(QtGui.QPixmap(image_file_path0))

		#Get system info
		available_cores, total_RAM, _ = parse_system_info()        
		assigned_cores = available_cores//2 #half
		self.assigned_RAM = total_RAM[1] #75%    

		#Adjust GUIs
		self.spinBox_5.setMaximum(available_cores)
		self.spinBox_5.setValue(assigned_cores)
		
		#Default inputs (edit manually)		
		self.lineEdit_7.setText("trial_1")			
		self.lineEdit_6.setText(r"E:\Teresa_article collab\91702_80R6w\tiff")		
		self.lineEdit_5.setText("91702-(.+)")
		self.lineEdit_4.setText("tag_1")	
		#radio buttons		
		self.option2 = 0 #flip ud
		self.option1 = 0 #recoloured
		self.option3 = 'linear' #input types
		self.radioButton_6.setEnabled(False)
		#checkboxes				
		self.items_output1 = ['linear'] #, 'log'
		self.items_output2 = ['pca']
		self.checkBox.setEnabled(False)		
		#widget list
		self.list_widget = []		
		#comboxes
		self.setup_combobox_data()		
		#execution button
		self.pushButton_7.setEnabled(False)

		#Define functionality     
		#1
		self.pushButton.clicked.connect(self.open_folder_dialog)		
		#2
		self.pushButton_4.clicked.connect(self.refresh_files)			
		self.Add.clicked.connect(self.browse_files)			
		self.Remove.clicked.connect(self.remove_selected_item)
		self.Clear.clicked.connect(self.remove_all_items)	
		self.toolButton.clicked.connect(self.move_item_up)
		self.toolButton_2.clicked.connect(self.move_item_down)			
		#load models
		self.pushButton_2.clicked.connect(self.open_file_dialog1) 
		self.pushButton_3.clicked.connect(self.open_file_dialog2)		
		self.pushButton_5.clicked.connect(self.open_file_dialog3)	
		#3		
		self.pushButton_7.clicked.connect(self.runningFunction) 		 
		
		#Build input lists, connect stateChanged signal to a common handler
		#1
		self.checkBox.stateChanged.connect(lambda state, item="linear": self.update_list(state, item))
		self.checkBox_2.stateChanged.connect(lambda state, item="log": self.update_list(state, item))		
		self.checkBox_3.stateChanged.connect(lambda state, item="original": self.update_list(state, item))
		
		self.checkBox_2.stateChanged.connect(self._on_checkbox_state_changed2)

		#2
		self.checkBox_5.stateChanged.connect(lambda state, item="pca": self.update_list2(state, item))
		self.checkBox_4.stateChanged.connect(lambda state, item="dsa": self.update_list2(state, item))
		self.checkBox_6.stateChanged.connect(lambda state, item="umap": self.update_list2(state, item))

	#endregion 

	#Disabling Run button
	def update_button_state(self):		
		if self.listWidget.count() == 0:
			self.pushButton_7.setEnabled(False)
		else:
			self.pushButton_7.setEnabled(True)

	#region (1) functions  

	def open_folder_dialog(self):
		# Open the folder selection dialog
		folder_path = QFileDialog.getExistingDirectory(
			parent=self,                                      # Parent widget
			caption="Select Input Folder",                     # Dialog title
			directory="",                          # Initial directory QDir.currentPath()
			options=QFileDialog.Option.ShowDirsOnly
			) #Filter		

		if folder_path: # If a folder was selected (user didn't cancel)
			# Display the selected folder path (e.g., in a QLineEdit)
			self.lineEdit_6.setText(folder_path)  

	#Rescaling
	def update_list(self, state, item_value):
		if state == Qt.Checked:
			if item_value not in self.items_output1:
				self.items_output1.append(item_value)
		else: 
			if item_value in self.items_output1:
				self.items_output1.remove(item_value)  
	
	#Image processing
	def get_selected_option2(self): #flip ud
		if self.radioButton_4.isChecked():			
			self.option2 = 1
		elif self.radioButton_3.isChecked():			
			self.option2 = 0		
		else:			
			self.option2 = None				

	def get_selected_option(self): #save recoloured
		if self.radioButton.isChecked():			
			self.option1 = 1
		elif self.radioButton_2.isChecked():			
			self.option1 = 0		
		else:			
			self.option1 = None				
	
	def _on_checkbox_state_changed2(self, state):
		if state == Qt.Unchecked:
			self.radioButton_6.setEnabled(False)
			self.radioButton_5.setChecked(True)			
		else:			
			self.radioButton_6.setEnabled(True)

	#endregion

	#region (2) functions

	def get_selected_option3(self): #input type
		if self.radioButton_5.isChecked():			
			self.option3 = 'linear'
		elif self.radioButton_6.isChecked():			
			self.option3 = 'log'		
		else:			
			self.option3 = None			

	#listWidget
	def move_item_up(self):
		current_row = self.listWidget.currentRow()
		if current_row > 0:
			current_item = self.listWidget.takeItem(current_row)
			self.listWidget.insertItem(current_row - 1, current_item)
			self.listWidget.setCurrentRow(current_row - 1)

	def move_item_down(self):
		current_row = self.listWidget.currentRow()
		if current_row < self.listWidget.count() - 1:
			current_item = self.listWidget.takeItem(current_row)
			self.listWidget.insertItem(current_row + 1, current_item)
			self.listWidget.setCurrentRow(current_row + 1)		

	def remove_all_items(self):
		self.listWidget.clear()
		self.list_widget = []	

		self.update_button_state() #update

	def remove_selected_item(self):
		selected_item = self.listWidget.currentItem()
		if selected_item:
			row = self.listWidget.row(selected_item)
			removed_item = self.listWidget.takeItem(row)
			del removed_item		
		
		self.update_button_state() #update

	def refresh_files(self):
		
		trial_name = self.lineEdit_7.text()		
		imageFolder = self.lineEdit_6.text() #add quotes	 		
		input_expression = self.lineEdit_5.text()
		extension = self.comboBox_2.currentText()
		tag_combo = self.lineEdit_4.text()
		input_type = self.option3 #linear, log		
		type_list = self.items_output1
		
		workingDir = os.path.join(imageFolder, trial_name)	
		outputFolder = os.path.join(workingDir, 'transformation_results\\'+  tag_combo)  		
		mkdir2(workingDir) #keeper	
		make_dir(outputFolder) #multiple		

		#Build list
		fileList0, pattern = getFileList(imageFolder, input_expression, extension)
		path_table = build_filePaths(fileList0, pattern, outputFolder)		
		file_paths = path_table["path"].tolist()
		
		if input_type in type_list:			

			# Add selected file paths to QListWidget
			temp_list = self.list_widget						
			temp_list_new = []
			if file_paths:			
				for path in file_paths:				
					if path in temp_list:											
						continue				
					else:
						temp_list_new.append(path)												
				
			temp_list.extend(temp_list_new)
			
			self.list_widget = temp_list
			self.listWidget.addItems(temp_list_new)	

		else:
			print(f"Tick on {input_type} tile export checkbox.")			

		self.update_button_state() #update


	def browse_files(self):			
		defaultFolder = self.lineEdit_6.text()
		print(defaultFolder)
		#Open dialog
		file_dialog = QFileDialog()
		file_paths, _ = file_dialog.getOpenFileNames(
			self,
			"Select Files", #Title
			defaultFolder, #starting dir
			"All Files (*);;") #Filter

		# Add selected file paths to QListWidget
		temp_list = self.list_widget						
		temp_list_new = []
		if file_paths:			
			for path in file_paths:				
				if path in temp_list:											
					continue				
				else:
					temp_list_new.append(path)												
			
		temp_list.extend(temp_list_new)
		
		self.list_widget = temp_list
		self.listWidget.addItems(temp_list_new)

		self.update_button_state() #update

	#Load models
	def open_file_dialog1(self):
		defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;PCA model (*.pkl)" # File filters
		)
		if file_path:
			self.lineEdit_8.setText(file_path)	

	def open_file_dialog2(self):
		defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;DSA model (*.tar)" # File filters
		)
		if file_path:
			self.lineEdit_9.setText(file_path)	
	
	def open_file_dialog3(self):
		defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;UMAP model (*.xml)" # File filters
		)
		if file_path:
			self.lineEdit_10.setText(file_path)	

	#Dimensionality reduction
	def update_list2(self, state, item_value):
		if state == Qt.Checked:
			if item_value not in self.items_output2:
				self.items_output2.append(item_value)
		else:
			if item_value in self.items_output2:
				self.items_output2.remove(item_value)	

   	#endregion	
	   
	#region (3) functions

	#prediction output resolution 
	def setup_combobox_data(self):		
		self.comboBox_5.setItemData(0, 1)
		self.comboBox_5.setItemData(1, 2)
		self.comboBox_5.setItemData(2, 4)
		self.comboBox_5.setItemData(3, 8)
		self.comboBox_5.setItemData(4, 16)
		self.comboBox_5.setItemData(5, 32)
		self.comboBox_5.setItemData(6, 64)
		

	#endregion
	
	#region Main script (execution)	
	
	def runningFunction(self):        	

		#User input
		#1
		trial_name = self.lineEdit_7.text() #to prevent overwriting
		imageFolder = self.lineEdit_6.text()		
		tileSize = int(self.comboBox_3.currentText())
		type_list = self.items_output1		
		pctOut = self.doubleSpinBox_2.value() #percentile out in the input (for colour contrast)
		filterSize = self.spinBox_6.value() #default=5; for smoothness
		flip_ud = self.option2 #flip upside-down (=1 if ImageJ differs from Windows viewer)		
		save_recoloured = self.option1 #save chemical elements heatmap

		#2		
		tag_combo = self.lineEdit_4.text()
		input_type = self.option3 #radio button: linear, log 		
		transform_list = self.items_output2 #checkbox: pca, dsa, umap		
		model_path_pca = self.lineEdit_8.text() #previously saved (use if same input list)
		model_path_dsa = self.lineEdit_9.text() 
		model_path_umap = self.lineEdit_10.text()
		outPct = self.doubleSpinBox_3.value() #default= 0.03; percentile out in the output (for colour contrast)

		#3				
		nodes_layer1 = self.spinBox.value() #DSA network encoder
		nodes_layer2 = self.spinBox_3.value()
		alpha_reg = float(self.lineEdit_2.text()) #default= 0.001
		betha_reg = float(self.lineEdit.text()) #default= 10
		LEARNING_RATE = float(self.lineEdit_3.text()) #5e-2 (might diverge!, decrease if necessary)   
		BATCH_SIZE = int(self.comboBox.currentText()) #predict uses=8192 (recommended for training)   		
		epoch_default = self.spinBox_4.value() #default= 5, can get better Loss
		test_ratio = self.doubleSpinBox.value() #test/training ratio
		
		n_neighbours = self.spinBox_8.value() #UMAP
		min_dist = self.doubleSpinBox_6.value()
		
		fraction_user_pca = self.doubleSpinBox_4.value() #sub-sampling WSI
		fraction_user_dsa = self.doubleSpinBox_5.value()		
		fraction_user_umap = self.doubleSpinBox_7.value()
		scale = int(self.comboBox_4.currentText())
		resolution = int(self.comboBox_5.currentData())
		n_workers = self.spinBox_5.value() #half of available cores

		#Default	
		workingDir = os.path.join(imageFolder, trial_name) #Ensure there is a folder when running	
		outputFolder = os.path.join(workingDir, 'transformation_results\\'+  tag_combo)  		
		mkdir2(workingDir) #keeper
		make_dir(outputFolder) #or empty with make_dir  			

		chosen_table = qListWidget_list(self.listWidget, outputFolder)
		n_layers_input = chosen_table.shape[0]
		n_channels_bottleneck = 3 #default = 3 		
		network_nodes = [n_layers_input, nodes_layer1, nodes_layer2, n_channels_bottleneck] #15, 10, 3 (encoder default)    
		ADD_SPARSITY = 'yes' #if 'yes', the cost function is regularized    
		RHO = 0.5 #KL; default 0.5 (larger reduces train loss >> validation loss)
		BATCH_SIZE_pred = int(BATCH_SIZE/4) #e.g.: 2048 or 8192 (depends on tile size)   		
		n_workers_dsa = n_workers
		n_workers_umap = n_workers
		fileSize_th = 12 #default = 4GB; depends on PC				
		device = get_device() #'cpu', 'cuda:0'  
		# device = torch.device("cpu") #forced
		
		#Pre-calculated models		
		model_paths = [model_path_pca, model_path_umap, model_path_dsa]
		boolean_list = [os.path.exists(path) for path in model_paths]		
		true_indices = [i for i, x in enumerate(boolean_list) if x]
		condition_models = any(boolean_list) #condition_pca | condition_umap | condition_dsa
		condition_pca = boolean_list[0]
		condition_umap = boolean_list[1]
		condition_dsa = boolean_list[2]
	
		#Generate pyramids	
		print('Generating pyramids..')	
		if not condition_models:
			print('Calculating histogram limits..')
			_, _ = build_pyramids(chosen_table, type_list, tileSize, filterSize, 
						 pctOut, workingDir, save_recoloured, flip_ud)
			#metadata, descriptiveStats

			statsFile = os.path.join(workingDir, 'descriptiveStats.csv')	
		else:
			#prior histogram limits
			print('Loading prior histogram limits..')
			model_path = model_paths[true_indices[0]] #first (assumming all within same Trial folder)
			path1 = os.path.dirname(model_path)
			path2 = os.path.dirname(path1)
			workingDir_prior = os.path.dirname(path2)

			_, _ = build_pyramids_prior(chosen_table, type_list, tileSize, filterSize, 
							   workingDir_prior, workingDir, save_recoloured, flip_ud)

			statsFile = os.path.join(workingDir_prior, 'descriptiveStats.csv')	

		#Retrieve pyvips process metadata							
		descriptiveStats = pd.read_csv(statsFile)
		item1 = f"{input_type}_mean"
		item2 = f"{input_type}_stddev"
		scaler_input = descriptiveStats[[item1, item2]]	

		metadataFile = os.path.join(workingDir, 'tileConfiguration.csv')   				
		metadata = pd.read_csv(metadataFile)				
		
		#Learn about montage
		metadata2 = metadata.loc[metadata["type"] == input_type, :]   
		fileList = metadata2['filepath']		

		imageHeight = metadata2.loc[metadata2['x'] == 0, :]['H'].sum()
		imageWidth = metadata2.loc[metadata2['y'] == 0, :]['W'].sum()    
		tiles_across = metadata2.iloc[-1, :]['x'] + 1
		# tiles_height = metadata2.iloc[-1, :]['y'] + 1		
		dim = [imageWidth, imageHeight, n_layers_input]

		fraction_max = predict_fileSize(dim, fileSize_th) #1; fraction of data allowed for training     
		fraction_pca = min(fraction_user_pca, fraction_max)
		fraction_dsa = min(fraction_user_dsa, fraction_max)
		fraction_umap = min(fraction_user_umap, fraction_max) #7 min for 6013x4201 with 512x512 tiles		
		
		#Load models
		
		#Principal Component Analysis (PCA)
		if 'pca' in transform_list:									
			
			#Destination directory
			outputFolder2 = os.path.join(outputFolder, 'pca_tiles')   			
			mkdir2(outputFolder2)			

			if not condition_pca:		    
				print('PCA factorisation..')			

				modelPATH = incremental_PCA(fileList, scale, fraction_pca, scaler_input, outputFolder)  
				
			else:		
				print('Loading PCA model..')				

				#prior histogram limits
				path1 = os.path.dirname(model_path_pca) #tag_1
				path2 = os.path.dirname(path1) #transformation_results
				path3 = os.path.dirname(path2) #trial
				statsFile = os.path.join(path3, 'descriptiveStats.csv')#chosen_table must match
				descriptiveStats = pd.read_csv(statsFile)
				col1 = f"{input_type}_mean"
				col2 = f"{input_type}_stddev"
				scaler_input = descriptiveStats[[col1, col2]] #overwrite variable

				modelPATH = model_path_pca 

			print('PCA transformation..')
			fileList2 = transform_tiles_pca(fileList, resolution, scaler_input, modelPATH, 
								   outputFolder2, n_workers)
			
			print('Stitching..')
			if not condition_pca:			
				stitch_crop_rescale(fileList2, tiles_across, outPct, outputFolder)   	 
			else:
				stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder)   	 

		#Uniform Manifold Approximation and Projection (UMAP)
		if 'umap' in transform_list:		

			#Destination directory				
			outputFolder2 = os.path.join(outputFolder, 'umap_tiles')   			
			mkdir2(outputFolder2)
			
			if not condition_umap:	
				print('UMAP fitting..')		

				_, model_path = incremental_loading_UMAP(fileList, scale, fraction_umap, scaler_input,
										 n_neighbours, min_dist, outputFolder)  
				
				modelPATH = model_path
			else:	
				print('Loading UMAP model..')				

				#prior histogram limits
				path1 = os.path.dirname(model_path_umap)
				path2 = os.path.dirname(path1)
				path3 = os.path.dirname(path2)
				statsFile = os.path.join(path3, 'descriptiveStats.csv')#chosen_table must match
				descriptiveStats = pd.read_csv(statsFile)
				col1 = f"{input_type}_mean"
				col2 = f"{input_type}_stddev"
				scaler_input = descriptiveStats[[col1, col2]] #overwrite variable

				modelPATH = model_path_umap 

			print('UMAP transformation..')
			fileList2 = transform_tiles_umap(fileList, resolution, scaler_input, modelPATH, 
									outputFolder2, n_workers_umap)
			
			print('Stitching..')
			if not condition_umap:			
				stitch_crop_rescale(fileList2, tiles_across, outPct, outputFolder)   	 
			else:
				stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder)   
		
		#Deep sparse autoencoder (DSA)		
		if 'dsa' in transform_list:			
			outputFolder2 = os.path.join(outputFolder, 'dsa_tiles')               			
			mkdir2(outputFolder2)			
			
			if not condition_dsa:    
				print('DSA training..')				
				dataset_list = incremental_loading_DSA(fileList, scale, fraction_dsa, test_ratio)    

				incremental_training_DSA(dataset_list, BATCH_SIZE, network_nodes, LEARNING_RATE, epoch_default, 
										ADD_SPARSITY, alpha_reg, betha_reg, RHO, device, n_workers, outputFolder)
				
				modelPATH = f"{outputFolder}/model_epochs{epoch_default}.tar" #must be *.tar not *.pth    
			else:
				modelPATH = model_path_dsa   
			
			print('DSA transformation..')
			fileList2 = transform_tiles_dsa(fileList, resolution, modelPATH, network_nodes, 
											BATCH_SIZE_pred, device, n_workers_dsa, outputFolder2)
			
			print('Stitching..')
			if not condition_dsa:			
				stitch_crop_rescale(fileList2, tiles_across, outPct, outputFolder)   	 
			else:
				stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder)   
			

		print('Finished.')

	#endregion	

	#region Application
def custom_exception_handler(exc_type, exc_value, exc_traceback):		

	print(f"Unhandled exception caught: {exc_type.__name__}: {exc_value}")
	
	traceback.print_exception(exc_type, exc_value, exc_traceback)
	formatted_traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)	
	traceback_string = "".join(formatted_traceback_lines)

	# Display a message box to the user
	msg_box = QMessageBox()
	msg_box.setIcon(QMessageBox.Critical)
	msg_box.setWindowTitle("Error")
	msg_box.setText("An unexpected error occurred. Please, check inputs")
	msg_box.setInformativeText(traceback_string) #str(exc_value)
	
	msg_box.exec_()

if __name__ == "__main__":		
	
	#Dependencies
	import multiprocessing
	multiprocessing.freeze_support() #mandatory
	#(anything before prints for each core)
	
	import os   
	import sys	
	import traceback
	import pandas as pd    				

	#relative paths
	from helperFunctions.mkdir_options import make_dir, mkdir2	
	from main_functions import parse_system_info, get_device
	from functions_pyramids import qListWidget_list, getFileList, build_filePaths, build_pyramids, build_pyramids_prior, predict_fileSize, stitch_crop_rescale, stitch_crop_rescale_prior
	from functions_PCA import incremental_PCA, transform_tiles_pca
	from functions_DSA import incremental_loading_DSA, incremental_training_DSA, transform_tiles_dsa    	
	from functions_UMAP import incremental_loading_UMAP, transform_tiles_umap
	
	#GUI
	from PyQt5.QtWidgets import QApplication, QFileDialog	
	from PyQt5.QtCore import Qt
	from PyQt5 import QtGui		

	sys.excepthook = custom_exception_handler

	#Run
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())
	
	#endregion