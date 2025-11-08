# Chemistry Simplifier
# The package to process rock chemical maps as dimensionally reduced flat images in RGB while maximing the original images content of any micro-analysis technique.

**Version**: 1 (beta)  
**Author**: Dr Marco Acevedo Z. (maaz.geologia@gmail.com)  
**Affiliation**: School of Earth and Atmospheric Sciences, Queensland University of Technology  
**Date**: November 2025  
**Citation**: [Acevedo Zamora et al. 2024](https://www.sciencedirect.com/science/article/pii/S0009254124000779#f0015)  
**Previous versions**: [Original repository](https://github.com/marcoaaz/AcevedoEtAl._2024b_autoencoder)  

---

## 📖 Overview

Chemistry simplifier allows researchers to do pixel-based image analysis of full resolution micro-analysis maps (e.g., intensity from X-ray lines, mass spectrometry analytes) of any laboratory instrument into representation images using linear and non-linear dimensionality reduction. The available methods are Principal component analysis (PCA), the Uniform Manifold Approximation and Projection (UMAP), and the Symmetrical deep sparse autoencoder (DSA) neural network. The data-driven approach does not demand thinking on the input as much as reduction process parametrisation. 

Colourful image outputs are rapidly produced and are much simpler to understand than the separate original inputs since they are richer in information (texture). We call them pseudo-phase maps and they also are friendlier for image segmentation with [QuPath](https://qupath.github.io/) ([Bankhead et al., 2017](https://www.nature.com/articles/s41598-017-17204-5)) using the [pixel classifier](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html) tool. 

Locally, the output intermediate/final montages are saved in a structured folder sequence for each trial/tag combination (see interface). All processing metadata is recorded for potential future documentation in research papers. 

<img width=90% height=90% alt="Image" src="https://github.com/user-attachments/assets/be17dff8-6783-40f1-ac55-55009377d954" />


In well-characterised samples, we demonstrated that:
 - PCA and UMAP tends to focus more on mineralogy while DSA is attentive to compositional zonation within crystals
 - PCA is much faster and can more robust to noisy and/or artefact image inputs than DSA  
      
   

---

## 🚀 Features

### Core Functionality
- **Graphical User Interface (GUI) following three steps** for processing a large number of chemical maps with ease
- **High reliability and performance** due to parallelised implementation
- **Image file parsing** to find your experiment in an input folder using [regular expressions](https://docs.python.org/3/library/re.html)
- **Basic image processing** to adjust image contrast and flip input/outputs
- **Central image stack menu** to select/edit data inputs and models
- **Parametrisation and performance menus** for adjusting DSA and UMAP models, pixel sampling and input (training) and output (prediction) image size
   
### Adaptive Interface
- **Grid design** - adapts to the window size

---

## 🖥️ Requirements*

The current Chemistry simplifier version was demonstrated to work on Windows 11 OS.

- **Python** 3.9.13 for making "numba > llvm requirement" work
- **PyQt5** 5.15.11 for running GUI (designed with PyQt5-tools)
- **pyinstaller** 6.16.0 for compiling with modified generated main.spec file*
- **multiprocessing** (included with most Python installations) for parallel processing
- **pickle** (included with most Python installations) for saving scalers/models
- **Additional libraries**:
  - `pyvips 3.0.0`** - for enabling extreme processing speed with image pyramid processing [link](https://github.com/libvips/pyvips)
  - `umap-learn 0.5.9.post2` - for UMAP [link](https://umap-learn.readthedocs.io/en/latest/)
  - `numba 0.60.0` - for UMAP guts [link](https://numba.pydata.org/)
  - `torch 2.8.0+cu126` - for DSA [link](https://pytorch.org/get-started/locally/)
  - `scikit-learn 1.6.1` - for incremental PCA [link](https://scikit-learn.org)  

*Ensure the main.spec file contains:

    datas=[
        ("icons", "icons"),        
        ("c:/vips-dev-8.16/bin", "vips"),                
        ("E:/Alienware_March 22/current work/00-new code May_22/dimReduction_v2/chemSimplifier3/Lib/site-packages/llvmlite/*", "llvmlite"),        
        ("E:/Alienware_March 22/current work/00-new code May_22/dimReduction_v2/chemSimplifier3/Lib/site-packages/llvmlite/binding/*", "llvmlite/binding"),
        ],
    hiddenimports=['numba.core.runtime', 'numba.core.registry'],
        
- **pyvips requires internally defining the path to libvips binaries (Windows DLL) in your PC. I downloaded the folder from [link](https://github.com/libvips/build-win64-mxe/releases/tag/v8.16.0) and unzipped to 'c:/vips-dev-8.16/bin'



---

## 📁 Versions Available

### Cube converter v1 (main.py)

- File to call the functionality and app interface (cubeConverter_v3.py)
- Suitable for reading and processing VSI files (CellSense format) saved from Evident VS200 slide scanner (at QUT)
- All metadata extraction features included
  
---

## ⌨️ Creating the Executable

1.  In VSCode or Anaconda, create/activate <your-environment-name>
2.  **pip install -r requirements.txt**
2.  In the terminal, run:
   ```bash
   pyinstaller main.py
   ```
5.  Edit the main.spec file (see edits in Requirements section above)
   ```bash
   pyinstaller main.spec
   ```
6.  The executable will be generated next to a bundled app folder at:  
   "..\<your-environment-name>\dist\Chemistry Simplifier v1\Chemistry Simplifier v1.exe"


## 📦 Packaged Executable

- Chemistry simplifier v1.exe works for Windows 11 and it is not fully self contained (for efficiency while opening the app)
- A Terminal will be open to indicate the progress of processing your file
- An Error handling mechanism pops up if the user inputs a wrong value in the GUI options. For persistent errors, please, send me a screenshot

## Issues and future work

- This is a beta version that will soon be improved with user feedback
- You are welcome to reach out and share your developing ideas with me. Under a scientific collaboration project, I could help you design, implement, and trial new Cube converter software options.
- I had in mind:
  - Cloud implementation with more processing cores
  - Trialling UMAP more extensively as it is also more computationally expensive to use a neighbour node graph model
- Support for Mac OS and Linux 

## Related papers

The software depends on open-source, scientific citations and user feedback. The following research papers already have contributed to its evolution (directly or indirectly):

  - Acevedo Zamora, M. A., & Kamber, B. S. (2023). Petrographic Microscopy with Ray Tracing and Segmentation from Multi-Angle Polarisation Whole-Slide Images. Minerals, 13(2), 156. [https://doi.org/10.3390/min13020156](https://doi.org/10.3390/min13020156)
  - Acevedo Zamora, M. (2024). Petrographic microscopy of geologic textural patterns and element-mineral associations with novel image analysis methods [Thesis by publication, Queensland University of Technology]. Brisbane. [https://eprints.qut.edu.au/248815/](https://eprints.qut.edu.au/248815/)
  - Burke, T. M., Kamber, B. S., & Rowlings, D. (2025). Microscopic investigation of incipient basalt breakdown in soils: implications for selecting products for enhanced rock weathering [Original Research]. Frontiers in Climate, Volume 7 - 2025. [https://doi.org/10.3389/fclim.2025.1572341](https://doi.org/10.3389/fclim.2025.1572341)
 
  Thanks.  
  Marco
