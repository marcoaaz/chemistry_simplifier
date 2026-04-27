'''
main.py

Version 1 of software GUI processing multi-elemental chemical maps into Principal Component Analysis (PCA),
Deep Sparse Autoencoder (DSA), and Uniform Manifold Approximation and Projection whole-slide image representations.

Documentation:
https://www.youtube.com/watch?v=2EjrLpC4cE4&t=163s
https://pyinstaller.org/en/stable/usage.html

Written in python 3.9.13
environment: chemSimplifier3

Logo: https://stock.adobe.com/search?k=color+wheel+logo&asset_id=56781933

Created: 19-Sep-25, Marco Acevedo
Updated: 23-Sep-25, 30-Sep-25, 19-Feb-26, 28-Apr-26

Current: python 3.9.13 (chemSimplifier3)

'''

#Essential dependencies
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from chemistrySimplifier_v5 import Ui_MainWindow #relative path
from umap_window import Ui_umapWindow
from dsa_window import Ui_dsaWindow

#region Sub-Windows

class UmapWindow(QMainWindow, Ui_umapWindow):
	def __init__(self):
		super().__init__()
		
		self.setupUi(self)

		#Store
		self.saved_state = {
			"n_neighbours": self.spinBox_8.value(),
			"min_dist": self.doubleSpinBox_6.value(),
		}

		#Connect buttons to logic
		self.pushButton.clicked.connect(self.handle_accept)
		self.pushButton_2.clicked.connect(self.handle_cancel)

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Escape:
			self.handle_cancel()
		else:
			super().keyPressEvent(event)
			
	def handle_accept(self):
		self.saved_state["n_neighbours"] = self.spinBox_8.value()
		self.saved_state["min_dist"] = self.doubleSpinBox_6.value()

		self.hide()

	def handle_cancel(self):
		self.spinBox_8.setValue(self.saved_state["n_neighbours"])
		self.doubleSpinBox_6.setValue(self.saved_state["min_dist"])

		self.hide()

class DsaWindow(QMainWindow, Ui_dsaWindow):
	def __init__(self):
		super().__init__()
		
		self.setupUi(self)

		self.setup_combobox_data4()	#DSA optimiser

		#Store
		self.saved_state = {
			"test_ratio": self.doubleSpinBox.value(),
			"nodes_layer1": self.spinBox.value(),
			"nodes_layer2": self.spinBox_3.value(),
			"noise_value": self.doubleSpinBox_5.value(),
			"criterion_type": self.comboBox_22.currentData(),
			"alpha_reg": self.lineEdit_2.text(),
			"betha_reg": self.lineEdit.text(),
			"BATCH_SIZE": self.comboBox.currentText(),
			"LEARNING_RATE": self.lineEdit_3.text(),
			"epoch_default": self.spinBox_4.value(),
			}

		#Connect buttons to logic
		self.pushButton.clicked.connect(self.handle_accept)
		self.pushButton_2.clicked.connect(self.handle_cancel)

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Escape:
			self.handle_cancel()
		else:
			super().keyPressEvent(event)

	def handle_accept(self):		
		
		self.saved_state["test_ratio"] = self.doubleSpinBox.value() 		
		self.saved_state["nodes_layer1"] = self.spinBox.value() 
		self.saved_state["nodes_layer2"] = self.spinBox_3.value()
		self.saved_state["noise_value"] = self.doubleSpinBox_5.value() 
		self.saved_state["criterion_type"] = self.comboBox_22.currentData()
		self.saved_state["alpha_reg"] = self.lineEdit_2.text() 
		self.saved_state["betha_reg"] = self.lineEdit.text() 
		self.saved_state["BATCH_SIZE"] = self.comboBox.currentText()  		
		self.saved_state["LEARNING_RATE"] = self.lineEdit_3.text()   		
		self.saved_state["epoch_default"] = self.spinBox_4.value() 

		self.hide()

	def handle_cancel(self):		
		
		self.doubleSpinBox.setValue(self.saved_state["test_ratio"])
		self.spinBox.setValue(self.saved_state["nodes_layer1"])
		self.spinBox_3.setValue(self.saved_state["nodes_layer2"])
		self.doubleSpinBox_5.setValue(self.saved_state["noise_value"])

		index = self.comboBox_22.findData(self.saved_state["criterion_type"])
		self.comboBox_22.setCurrentIndex(index if index != -1 else 0)	
	
		self.lineEdit_2.setText(self.saved_state["alpha_reg"])
		self.lineEdit.setText(self.saved_state["betha_reg"])
		self.comboBox.setCurrentText(self.saved_state["BATCH_SIZE"])
		self.lineEdit_3.setText(self.saved_state["LEARNING_RATE"])
		self.spinBox_4.setValue(self.saved_state["epoch_default"])
		
		self.hide()

	#DSA optimiser 
	def setup_combobox_data4(self):		
		self.comboBox_22.setItemData(0, 'MSE')
		self.comboBox_22.setItemData(1, 'L1')

#endregion

#region Main GUI

class Window(QMainWindow, Ui_MainWindow):
	
	def __init__(self):
		super().__init__()
		self.setupUi(self)		

		#Keep track of child windows		
		self.umap_window = UmapWindow()
		self.dsa_window = DsaWindow()

		#Recovering images	
		# relative_path = sys._MEIPASS #PyInstaller executable
		bundle_dir = os.path.abspath(os.path.dirname(__file__)) #relative path
		gui_icon_path1 = os.path.join(bundle_dir, "icons/colour_wheel.ico")
		gui_icon_path2 = os.path.join(bundle_dir, "icons/gear.png")
		gui_image_path0 = os.path.join(bundle_dir, "icons/QUT-Logo.png")
		gui_image_path1 = os.path.join(bundle_dir, "icons/AuScope_logo.png")		
		
		#Window
		self.setWindowTitle("Chemistry Simplifier v1.2")
		self.setWindowIcon(QIcon(gui_icon_path1))
		self.setMinimumSize(600, 600)
		self.setWindowFlags(self.windowFlags()) 

		#GUI image/icon updates				
		self.label_41.setPixmap(QPixmap(gui_image_path0))		
		self.label_53.setPixmap(QPixmap(gui_image_path1))
		self.pushButton_9.setIcon(QIcon( QPixmap(gui_icon_path2) ))
		self.pushButton_9.setIconSize(QSize(16,16))
		self.pushButton_10.setIcon(QIcon( QPixmap(gui_icon_path2) ))
		self.pushButton_10.setIconSize(QSize(16,16))

		#Default inputs (edit manually)	
			
		self.lineEdit_7.setText("trial_1")			
		self.lineEdit_6.setText(r"E:\Teresa_article collab\91702_80R6w\tiff")		
		self.lineEdit_5.setText("(.+)")
		self.lineEdit_4.setText("tag_1")	
		#radio buttons		
		self.option3 = 'linear' #input types		
		self.option4 = 'standardising' #scaling types (normalisation)
		self.option1 = 1 #recoloured
		self.option2 = 0 #flip ud						
		#checkboxes				
		self.items_output1 = ['linear'] #linear, log
		self.items_output2 = ['pca']		
		self.items_output3 = [] #moving image
		#widget list
		self.list_widget = []		
		#update combo boxes
		self.setup_combobox_data() #prediction resolution		
		self.setup_combobox_data2()	#transformation model
		self.setup_combobox_data3()	#interpolation
		

		#Get system info
		available_cores, total_RAM, _ = parse_system_info()        
		assigned_cores = available_cores//2 #half
		self.assigned_RAM = total_RAM[1] #75%    

		#Adjust GUIs
		self.spinBox_5.setMaximum(available_cores)
		self.spinBox_5.setValue(assigned_cores)

		#Define functionality     
		#Build input lists, connect stateChanged signal to a common handler

		#(1)
		self.pushButton.clicked.connect(self.open_folder_dialog)

		self.checkBox.stateChanged.connect(lambda state, item="linear": self.update_list(state, item))
		self.checkBox_2.stateChanged.connect(lambda state, item="log": self.update_list(state, item))				
		self.checkBox_3.stateChanged.connect(lambda state, item="original": self.update_list(state, item))				

		self.checkBox_2.stateChanged.connect(self._on_checkbox_state_changed2)

		self.radioButton_7.toggled.connect(self.get_selected_option4) #normalisation

		#(2)		
		self.radioButton_5.toggled.connect(self.get_selected_option3) #use linear input
		
		self.listWidget.setSelectionMode(QAbstractItemView.ExtendedSelection) #Ctrl/Shift selection
		self.pushButton_4.clicked.connect(self.refresh_files)			
		self.Add.clicked.connect(self.browse_files)			
		self.Remove.clicked.connect(self.remove_selected_item)
		self.Clear.clicked.connect(self.remove_all_items)	
		self.toolButton.clicked.connect(self.move_item_up)
		self.toolButton_2.clicked.connect(self.move_item_down)	
		
		#select models
		self.checkBox_5.stateChanged.connect(lambda state, item="pca": self.update_list2(state, item))
		self.checkBox_4.stateChanged.connect(lambda state, item="dsa": self.update_list2(state, item))
		self.checkBox_6.stateChanged.connect(lambda state, item="umap": self.update_list2(state, item))

		#setup models
		self.pushButton_9.clicked.connect(self.openWindow_umap) #umap window		 	
		self.pushButton_10.clicked.connect(self.openWindow_dsa) #dsa window

		#load models
		self.pushButton_2.clicked.connect(self.open_file_dialog1) 
		self.pushButton_3.clicked.connect(self.open_file_dialog2)		
		self.pushButton_5.clicked.connect(self.open_file_dialog3)	

		#(3)	
		self.radioButton.toggled.connect(self.get_selected_option) #false colour images
		self.radioButton_4.toggled.connect(self.get_selected_option2) #flip upside down

		self.pushButton_8.clicked.connect(self.open_file_dialog4) #control points
		self.pushButton_11.clicked.connect(self.open_file_dialog6) #extra moving image
		self.pushButton_6.clicked.connect(self.open_file_dialog5) #fixed image
		self.lineEdit_11.textChanged.connect(self.validate_and_fill) #size; .editingFinished		
		
		self.checkBox_8.stateChanged.connect(lambda state, item="recoloured": self.update_list3(state, item))
		self.checkBox_9.stateChanged.connect(lambda state, item="represented": self.update_list3(state, item))
		self.checkBox_7.stateChanged.connect(lambda state, item="originals": self.update_list3(state, item))

		self.pushButton_7.clicked.connect(self.runningFunction) 

		#High-level enabling 
		self.radioButton_6.setEnabled(False)
		self.checkBox.setEnabled(False)				
		self.pushButton_7.setEnabled(False) #execution button			
		
	def keyPressEvent(self, event):
		# Check if Esc was pressed
		if event.key() == Qt.Key_Escape:			
			self.close() 
		else:			
			super().keyPressEvent(event)	

	#endregion 

	#region Extra windows functions		
	
	def closeEvent(self, event):
		"""This runs when the main window is closed."""
		# Force the entire application to shut down immediately
		QApplication.quit()	
		
		event.accept()
		
	#custom class, which already has setupUi() built-in
	def openWindow_umap(self):				
		self.umap_window.show()
		self.umap_window.move(100, 300)

		self.umap_window.raise_() #move to the top
		self.umap_window.activateWindow() #focus

	def openWindow_dsa(self):				
		self.dsa_window.show()		
		self.dsa_window.move(100, 600)

		self.dsa_window.raise_() #move to the top
		self.dsa_window.activateWindow() #focus

	#endregion

	#region (1) functions  

	def open_folder_dialog(self):
		#Get the current path from the lineEdit to use as a starting point
		last_path = self.lineEdit_6.text()		

		#Create the dialog instance
		dialog = QFileDialog(self)
		dialog.setWindowTitle("Select Input Folder")

		dialog.setFileMode(QFileDialog.Directory) # ExistingFile or Directory
		dialog.setOption(QFileDialog.DontUseNativeDialog, True) #medicine: Windows OS
		# dialog.setOption(QFileDialog.ShowDirsOnly, False) #prevent hiding
		
		if os.path.exists(last_path):
			if os.path.isdir(last_path):
				dialog.setDirectory(last_path)
			elif os.path.isfile(last_path):
				dialog.setDirectory(os.path.dirname(last_path))
		else:
			dialog.setDirectory(os.getcwd()) #Initial directory QDir.currentPath()

		#Execute and get the folder
		if dialog.exec_():
			selected = dialog.selectedFiles()[0]			
			
			if os.path.isdir(selected):
				folder_path = selected
			else:
				folder_path = os.path.dirname(selected)

			self.lineEdit_6.setText(folder_path)

	#Bit-depth scaling
	def update_list(self, state, item_value):
		if state == Qt.Checked:
			if item_value not in self.items_output1:
				self.items_output1.append(item_value)
		else: 
			if item_value in self.items_output1:
				self.items_output1.remove(item_value)  
	
	#Image processing
	def _on_checkbox_state_changed2(self, state):
		if state == Qt.Unchecked:
			self.radioButton_6.setEnabled(False)
			self.radioButton_5.setChecked(True)			
		else:			
			self.radioButton_6.setEnabled(True)

	def get_selected_option4(self): 
		if self.radioButton_7.isChecked():			
			self.option4 = 'standardising'
		elif self.radioButton_8.isChecked():			
			self.option4 = 'centering'		
		else:			
			self.option4 = None				
	
	#endregion

	#region (2) functions	
	
	def get_selected_option3(self): #input type
		if self.radioButton_5.isChecked():			
			self.option3 = 'linear'
		elif self.radioButton_6.isChecked():			
			self.option3 = 'log'		
		else:			
			self.option3 = None		

	#Disabling Run button
	def update_button_state(self):		
		if self.listWidget.count() == 0:
			self.pushButton_7.setEnabled(False)
		else:
			self.pushButton_7.setEnabled(True)	

	#listWidget
	def move_item_up(self):
		# Get selected rows and sort them (0, 1, 2...)
		selected_rows = sorted([self.listWidget.row(item) for item in self.listWidget.selectedItems()])
		
		# If the first selected row is already at 0, we can't move up
		if not selected_rows or selected_rows[0] == 0:
			return

		for row in selected_rows:
			item = self.listWidget.takeItem(row)
			self.listWidget.insertItem(row - 1, item)
			item.setSelected(True) # Keep the item selected after move

	def move_item_down(self):
		# Get selected rows and sort them in REVERSE (...2, 1, 0)
		selected_rows = sorted([self.listWidget.row(item) for item in self.listWidget.selectedItems()], reverse=True)
		
		# If the last selected row is at the bottom, we can't move down
		if not selected_rows or selected_rows[0] == self.listWidget.count() - 1:
			return

		for row in selected_rows:
			item = self.listWidget.takeItem(row)
			self.listWidget.insertItem(row + 1, item)
			item.setSelected(True)		
	
	def remove_selected_item(self):
		selected_items = self.listWidget.selectedItems()
		if not selected_items:
			return

		for item in selected_items:
			text = item.text()
			if text in self.list_widget:
				self.list_widget.remove(text) # Remove from the tracking list

			# We take the item by row index
			row = self.listWidget.row(item)
			self.listWidget.takeItem(row)
			del item # Explicitly free memory
		
		self.update_button_state() #update
	
	def remove_all_items(self):
		self.listWidget.clear()
		self.list_widget = []	

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
		defaultFolder = self.lineEdit_8.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;PCA model (*.pkl)" # File filters
			)
		if file_path:
			self.lineEdit_8.setText(file_path)	
	
	def open_file_dialog3(self):
		defaultFolder = self.lineEdit_10.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;UMAP model (*.xml)" # File filters
			)
		if file_path:
			self.lineEdit_10.setText(file_path)	

	def open_file_dialog2(self):
		defaultFolder = self.lineEdit_9.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;DSA model (*.tar)" # File filters
		)
		if file_path:
			self.lineEdit_9.setText(file_path)	

	#Dimensionality reduction
	def update_list2(self, state, item_value):
		if state == Qt.Checked:
			if item_value not in self.items_output2:
				self.items_output2.append(item_value)
		else:
			if item_value in self.items_output2:
				self.items_output2.remove(item_value)	

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
	   
	#region (3) functions	
	
	def get_selected_option(self): #save recoloured
		if self.radioButton.isChecked():			
			self.option1 = 1
		elif self.radioButton_2.isChecked():			
			self.option1 = 0		
		else:			
			self.option1 = None	

	def get_selected_option2(self): #flip ud
		if self.radioButton_4.isChecked():			
			self.option2 = 1
		elif self.radioButton_3.isChecked():			
			self.option2 = 0		
		else:			
			self.option2 = None			

	#control points
	def open_file_dialog4(self):
		defaultFolder = self.lineEdit_14.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()

		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*);;landmarks (*.csv)" # File filters
		)
		if file_path:
			self.lineEdit_14.setText(file_path)	

	#Choosing extra moving image
	def open_file_dialog6(self):
		defaultFolder = self.lineEdit_12.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()
			
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*)" # File filters
		)
		if file_path:
			self.lineEdit_12.setText(file_path)	

	#Choosing fixed image
	def open_file_dialog5(self):
		defaultFolder = self.lineEdit_11.text()

		if defaultFolder == "":
			defaultFolder = self.lineEdit_6.text()
			
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select File",
			defaultFolder,  # Initial directory (empty string for default)
			"All Files (*)" # File filters
		)
		if file_path:
			self.lineEdit_11.setText(file_path)	

	def validate_and_fill(self):
		
		filepath = self.lineEdit_11.text()
		
		if os.path.isfile(filepath):			
			size_fixed = read_img_size(filepath)			
			#update
			self.spinBox_7.setValue(size_fixed[0])
			self.spinBox_8.setValue(size_fixed[1])						
		else:			
			self.spinBox_7.setValue(0)
			self.spinBox_8.setValue(0)
			print("Invalid file path.")
			
	#Choosing moving image
	def update_list3(self, state, item_value):
		if state == Qt.Checked:
			if item_value not in self.items_output3:
				self.items_output3.append(item_value)
		else: 
			if item_value in self.items_output3:
				self.items_output3.remove(item_value) 

	def setup_combobox_data2(self):		
		self.comboBox_8.setItemData(0, 'similarity')
		self.comboBox_8.setItemData(1, 'affine')
		self.comboBox_8.setItemData(2, 'tps')		

	def setup_combobox_data3(self):		
		self.comboBox_9.setItemData(0, 'nearest')
		self.comboBox_9.setItemData(1, 'bilinear')
		self.comboBox_9.setItemData(2, 'bicubic')		

	#endregion
	
	#region Main script (execution)	
	
	def runningFunction(self):        	

		#User input
		#1
		imageFolder = self.lineEdit_6.text()				
		trial_name = self.lineEdit_7.text() #to prevent overwriting		
		tag_combo = self.lineEdit_4.text()

		tileSize = int(self.comboBox_3.currentText())
		type_list = self.items_output1		
		pctOut = self.doubleSpinBox_2.value() #percentile out in the input (for colour contrast)
		filterSize = self.spinBox_6.value() #default=5; for smoothness		
		scaling_type = self.option4 #normalisation		

		#2			
		input_type = self.option3 #radio button: linear, log 		
		imageRepresentation_list = self.items_output2 #checkbox: pca, dsa, umap		
		model_path_pca = self.lineEdit_8.text() #previously saved (use if same input list)
		model_path_umap = self.lineEdit_10.text()
		model_path_dsa = self.lineEdit_9.text() 		

		fraction_user_pca = self.doubleSpinBox_4.value() #sub-sampling WSI
		fraction_user_dsa = self.doubleSpinBox_5.value()		
		fraction_user_umap = self.doubleSpinBox_7.value()
		scale = int(self.comboBox_4.currentText())
		resolution = int(self.comboBox_5.currentData())

		#3				
		save_recoloured = self.option1 #save chemical elements heatmap		
		colormap_name = self.comboBox_6.currentText()
		outPct = self.doubleSpinBox_3.value() #default= 0.03; percentile out in the output (for colour contrast)
		flip_ud = self.option2 #flip upside-down (=1 if ImageJ differs from Windows viewer)		

		path_csv = self.lineEdit_14.text()		
		moving_image_list = self.items_output3
		moving_extra_path = self.lineEdit_12.text()
		t_w = int(self.spinBox_7.text())
		t_h = int(self.spinBox_8.text())
		
		transformation_method = self.comboBox_8.currentData()
		interpolation_method = self.comboBox_9.currentData()		

		n_workers = self.spinBox_5.value() #half of available cores

		#UMAP window
		n_neighbours = self.umap_window.spinBox_8.value() #UMAP
		min_dist = self.umap_window.doubleSpinBox_6.value()

		#DSA window		
		test_ratio = self.dsa_window.doubleSpinBox.value() #test/training ratio		
		nodes_layer1 = self.dsa_window.spinBox.value() #DSA network encoder
		nodes_layer2 = self.dsa_window.spinBox_3.value()
		noise_value = self.dsa_window.doubleSpinBox_5.value() 
		criterion_type = self.dsa_window.comboBox_22.currentData()
		alpha_reg = float(self.dsa_window.lineEdit_2.text()) #default= 0.001, weight decay term
		betha_reg = float(self.dsa_window.lineEdit.text()) #default= 10, sparsity term
		BATCH_SIZE = int(self.dsa_window.comboBox.currentText()) #predict uses=8192 (recommended for training)   		
		LEARNING_RATE = float(self.dsa_window.lineEdit_3.text()) #5e-2 (might diverge!, decrease if necessary)   		
		epoch_default = self.dsa_window.spinBox_4.value() #default= 5, can get better Loss		

		#Default	
		workingDir = os.path.join(imageFolder, trial_name) #Ensure there is a folder when running	
		outputFolder = os.path.join(workingDir, 'transformation_results\\'+  tag_combo) 		
		mkdir2(workingDir) #keeper (note: make less redundant in next version)
		make_dir(outputFolder) #when changing Tag folder		
		chosen_table = qListWidget_list(self.listWidget, outputFolder) #format/search 		

		n_layers_input = chosen_table.shape[0]
		n_channels_bottleneck = 3 #3 		
		network_nodes = [n_layers_input, nodes_layer1, nodes_layer2, n_channels_bottleneck] #15, 10, 3 (encoder default)    
		ADD_SPARSITY = 'yes' #if 'yes', the cost function is regularized    
		RHO = 0.5 #KL; 0.5 (larger reduces train loss >> validation loss)
		BATCH_SIZE_pred = int(BATCH_SIZE/4) #e.g.: 2048 or 8192 (depends on tile size)   		
		device = get_device() #'cpu', 'cuda:0'  		
		
		fileSize_th = 12 #4GB; depends on PC				
		resolution_fix = resolution/scale #follows QuPath convention
		n_workers_pca = n_workers
		n_workers_dsa = n_workers
		n_workers_umap = n_workers		
		
		#dimensionality reduction conditions
		model_paths = [model_path_pca, model_path_umap, model_path_dsa]
		boolean_list = [os.path.exists(path) for path in model_paths]		
		true_indices = [i for i, x in enumerate(boolean_list) if x]
		condition_models = any(boolean_list) #condition_pca | condition_umap | condition_dsa
		condition_pca = boolean_list[0]
		condition_umap = boolean_list[1]
		condition_dsa = boolean_list[2]

		#registration conditions
		condition_moving_extra = (moving_extra_path != "")
		condition_landmarks = (path_csv != "")
		condition_size = not((t_w == 0) | (t_h == 0))
		condition_registration = condition_landmarks & condition_size

		#region Image pyramids
		print('Generating pyramids..')	
		
		lut = build_lut(colormap_name) #'jet'

		if not condition_models:
			print('Calculating histogram limits..')

			#metadata, descriptiveStats, recoloured_list
			_, _, recoloured_list = build_pyramids(chosen_table, scale, flip_ud, type_list, pctOut, filterSize, 
						 save_recoloured, lut, tileSize, workingDir)					

			statsFile = os.path.join(workingDir, 'descriptiveStats.csv')	
		else:
			#prior histogram limits
			print('Loading prior histogram limits..')
			
			model_path = model_paths[true_indices[0]] #first (assumming all within same Trial folder)
			path1 = os.path.dirname(model_path)
			path2 = os.path.dirname(path1)
			workingDir_prior = os.path.dirname(path2)

			_, _, recoloured_list = build_pyramids_prior(chosen_table, scale, flip_ud, type_list, workingDir_prior, filterSize, 
							   save_recoloured, lut, tileSize, workingDir)

			statsFile = os.path.join(workingDir_prior, 'descriptiveStats.csv')	

		#endregion

		#region Dimensionality reduction

		#Retrieve pyvips process metadata							
		descriptiveStats = pd.read_csv(statsFile)
		item1 = f"{input_type}_mean" #radiobutton
		item2 = f"{input_type}_stddev"
		scaler_input = descriptiveStats[[item1, item2]]	

		#Learn about montage
		metadataFile = os.path.join(workingDir, 'tileConfiguration.csv')   				
		metadata = pd.read_csv(metadataFile)						
		tiles_metadata = metadata.loc[metadata["type"] == input_type, :]   			
		tiles_across = tiles_metadata.iloc[-1, :]['x'] + 1 #information for stitching	 	

		#Preventing RAM overload (note: improve in next version..)		
		fraction_max = predict_datasetSize(tiles_metadata, n_layers_input, scale, fileSize_th, 32) #1; fraction of data allowed for training     
		fraction_pca = min(fraction_user_pca, fraction_max)
		fraction_dsa = min(fraction_user_dsa, fraction_max)
		fraction_umap = min(fraction_user_umap, fraction_max) #7 min for 6013x4201 with 512x512 tiles		
					
		#Load models		
		
		active_representation = [t for t in ['pca', 'umap', 'dsa'] if t in imageRepresentation_list]
		representation_list = [] #for registration

		for transform in active_representation:
			#Setup Folders
			outputFolder2 = os.path.join(outputFolder, f'{transform}_tiles')
			mkdir2(outputFolder2)

			#Fit models

			#Principal Component Analysis (PCA)
			if transform == 'pca':				
				is_prior = condition_pca	
				if not is_prior:		    
					print('PCA factorisation..')			
					modelPATH = incremental_PCA(tiles_metadata, fraction_pca, scaler_input, scaling_type, outputFolder)  
					
				else:		
					print('Loading previous PCA model..')				

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
				fileList2 = transform_tiles_pca(tiles_metadata, resolution_fix, scaler_input, scaling_type, modelPATH, 
									outputFolder2, n_workers_pca)				  	 

			#Uniform Manifold Approximation and Projection (UMAP)
			elif transform == 'umap':				
				is_prior = condition_umap
				if not is_prior:	
					print('UMAP fitting..')							
					_, modelPATH = incremental_loading_UMAP(tiles_metadata, fraction_umap, scaler_input, scaling_type,
											n_neighbours, min_dist, outputFolder)  					
					
				else:	
					print('Loading previous UMAP model..')				

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
				fileList2 = transform_tiles_umap(tiles_metadata, resolution_fix, scaler_input, scaling_type, modelPATH, 
										outputFolder2, n_workers_umap)			
			
			#Deep sparse autoencoder (DSA)		
			elif transform == 'dsa':			
				is_prior = condition_dsa
				if not is_prior:    
					print('DSA training..')				
					dataset_list = incremental_loading_DSA(tiles_metadata, fraction_dsa, test_ratio)    

					modelPATH = incremental_training_DSA(dataset_list, network_nodes, noise_value, criterion_type, BATCH_SIZE, LEARNING_RATE, epoch_default, 
											ADD_SPARSITY, alpha_reg, betha_reg, RHO, device, n_workers, outputFolder)					

				else:
					print('Loading previous DSA model..')	
					modelPATH = model_path_dsa   
				
				print('DSA transformation..')
				fileList2 = transform_tiles_dsa(tiles_metadata, resolution_fix, modelPATH, network_nodes,
									BATCH_SIZE_pred, device, n_workers_dsa, outputFolder2)		
			
			print(f'Stitching {transform} montage...')
			if not is_prior:
				representation_path = stitch_crop_rescale(fileList2, tiles_across, outPct, outputFolder)
			else:
				representation_path = stitch_crop_rescale_prior(fileList2, tiles_across, modelPATH, outputFolder)

			representation_list.append(representation_path)

		#endregion

		#region Image registration			
							
		if condition_registration:			
			print(f'Aligning montages...')	
			
			size_fixed = [t_w, t_h]

			registration_run(path_csv, size_fixed, moving_image_list, 
						recoloured_list, representation_list, chosen_table,  
						transformation_method, interpolation_method, outputFolder)

		if condition_moving_extra and os.path.exists(moving_extra_path):
			size_fixed = [t_w, t_h]

			#Setup Folders
			new_folder = f'registration_extra'
			outputFolder2 = os.path.join(outputFolder, new_folder)
			mkdir2(outputFolder2)        
			print(f"Saving into {new_folder}")

			input_basename = os.path.splitext(os.path.basename(moving_extra_path))[0]
			output_path = os.path.join(outputFolder2, f'{input_basename}_{transformation_method}.tif')

			register_wsi(moving_extra_path, size_fixed, path_csv, output_path, 
							transformation_method, interpolation_method)

		#endregion

		print('Finished.')

	#endregion	

	#region Application
def custom_exception_handler(exc_type, exc_value, exc_traceback):		

	print(f"Unhandled exception caught: {exc_type.__name__}: {exc_value}")
	
	traceback.print_exception(exc_type, exc_value, exc_traceback)
	formatted_traceback_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)	
	traceback_string = "".join(formatted_traceback_lines)

	# Display a message box
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
	from main_functions import parse_system_info, get_device, getFileList, build_filePaths, qListWidget_list, predict_datasetSize 	
	from functions_pyramids import build_lut, build_pyramids, build_pyramids_prior, stitch_crop_rescale, stitch_crop_rescale_prior	
	from functions_PCA import incremental_PCA, transform_tiles_pca
	from functions_DSA import incremental_loading_DSA, incremental_training_DSA, transform_tiles_dsa    	
	from functions_UMAP import incremental_loading_UMAP, transform_tiles_umap
	from functions_imageRegistration import read_img_size, registration_run, register_wsi
	
	#GUI
	from PyQt5.QtWidgets import QApplication, QFileDialog,  QAbstractItemView	
	from PyQt5.QtCore import Qt, QSize		
	from PyQt5.QtGui import QPixmap, QIcon
	

	sys.excepthook = custom_exception_handler

	#App resolution
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) 
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #icons
	
	if os.name == 'nt': #blurry on HDMI screens (Windows-only)
		import ctypes
		ctypes.windll.shcore.SetProcessDpiAwareness(1)
	
	#Run
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())
	
	#endregion