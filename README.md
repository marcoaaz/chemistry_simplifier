# Chemistry Simplifier
# The package to process rock chemical maps as dimensionally reduced flat images in RGB while maximing the original images content of any micro-analysis technique.

**Version**: 1.2  
**Binary download**: [Windows 11](https://zenodo.org/records/19688702)  
**Author**: Dr Marco Acevedo Z. (maaz.geologia@gmail.com)  
**Affiliation**: School of Earth and Atmospheric Sciences, Queensland University of Technology  
**Date**: 28-April-2026  
**Citation**: [Acevedo Zamora et al. 2024](https://www.sciencedirect.com/science/article/pii/S0009254124000779#f0015)  
**Original scripts**: [old repository](https://github.com/marcoaaz/AcevedoEtAl._2024b_autoencoder)  

---

## 📖 Overview

Chemistry simplifier software allows geologists to do pixel-based image analysis of full resolution micro-analysis maps (e.g., intensity from X-ray lines, mass spectrometry analytes) of any laboratory instrument into representation images using linear and non-linear dimensionality reduction. The available methods are Principal component analysis (PCA), the Uniform Manifold Approximation and Projection (UMAP), and the Deep sparse autoencoder (DSA) neural network. The data-driven approach focuses on the reduction process parametrisation and lesens the focus on interpreting the input images. 

Colourful image outputs are rapidly produced and are much simpler to understand than the separate original inputs since they are richer in information (texture). We call them pseudo-phase maps and they can be friendlier for image segmentation with pixel classification of multi-channel images in [QuPath](https://qupath.github.io/) ([Bankhead et al., 2017](https://www.nature.com/articles/s41598-017-17204-5)) using the [Pixel Classifier](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html) and [Image Combiner Warpy](https://github.com/BIOP/qupath-biop-catalog). 

Locally, the output intermediate/final montages are saved in a structured folder sequence for each trial/tag combination (see interface). All processing metadata is recorded for potential future documentation in research papers. 

<p align="center">
 <img width=80% height=80% alt="Image" src="https://github.com/user-attachments/assets/57b38ac7-201c-4e43-80df-c3c4baed6974" />
</p>

With trialling and well-characterised samples, I demonstrated that:
 - PCA can more robust to noisy and/or artefact image inputs than DSA. Therefore, PCA is the default choice (for initial assessment and image registration).  
 - PCA tends to highlight mineralogy while DSA is attentive to mineralogy and their compositional zonation within crystals. 
 - UMAP is good at distinguishing mineralogy and superior at denoising the output (grain/zone boundaries) than PCA because it is a non-linear method.
 - UMAP slows down when fitting the manifold to millions of pixels, making it slower than PCA and DSA.  

The image alignment options require control points that have been placed manually using ImageJ BigWarp plugin ([Bogovic et al., 2016](https://imagej.net/plugins/bigwarp)). The plugin allows exporting a 'landmarks.csv' file containing the ID and locations (X, Y) of the placemarks accross the moving and fixed images (at least 4 points are required to fit the transform models). This repository contains an example CSV to show the user for required format (see 'landmarks_bse_xpl.csv').

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
        
- **pyvips** requires internally defining the path to libvips binaries (Windows DLL) in your PC. I downloaded the folder from [link](https://github.com/libvips/build-win64-mxe/releases/tag/v8.16.0) and unzipped to 'c:/vips-dev-8.16/bin'



---

## 📁 Versions Available

### Chemistry Simplifier v1 (main.py)

- Suitable for reading and processing chemical maps from any spectroscopy and microanalysis technique as long as they are saved as images (TIF, JPEG, PNG, etc.)
- The internal process is recorded in metadata (fitted models, scaling parameters, element lists) for reproducibility.
  
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

- Chemistry simplifier v1.exe works for Windows 11 and it is not fully self contained (for efficiency while opening the app).
- A Terminal will be open next to the main window to indicate the progress of processing your file.
- An Error handling mechanism pops up if the user inputs a wrong value in the GUI options. For persistent errors, please, send me a screenshot.

## Issues and future work

This version can be improved with user feedback. You are welcome to reach out and share your developing ideas with me. Under a scientific collaboration project, I could help you design, implement, and trial new Cube converter software options.

- I had in mind:
  - Cloud implementation with more processing cores.
  - A checkpoint system within the GUI and process metadata to avoid recalculating the Trial folder intermediate steps (pyramid tiles, dimensionality reduction tiles)  

I would welcome support for Mac OS and Linux following the original Windows version. If interested, please let me know. 

## Related papers

The software depends on open-source, scientific citations and user feedback. The following research papers already have contributed to its evolution (directly or indirectly):

  - Acevedo Zamora, M. A., Kamber, B. S., Jones, M. W. M., Schrank, C. E., Ryan, C. G., Howard, D. L., Paterson, D. J., Ubide, T., & Murphy, D. T. (2024). Tracking element-mineral associations with unsupervised learning and dimensionality reduction in chemical and optical image stacks of thin sections. Chemical Geology, 650, 121997. https://doi.org/10.1016/j.chemgeo.2024.121997
  - Acevedo Zamora, M. (2024). Petrographic microscopy of geologic textural patterns and element-mineral associations with novel image analysis methods [Thesis by publication, Queensland University of Technology]. Brisbane. https://eprints.qut.edu.au/248815/
  - Ubide, T., Murphy, D. T., Emo, R. B., Jones, M. W. M., Acevedo Zamora, M. A., & Kamber, B. S. (2025). Early pyroxene crystallisation deep below mid-ocean ridges. Earth and Planetary Science Letters, 663, 119423. https://doi.org/10.1016/j.epsl.2025.119423 

 

 
  Thanks.  
  Marco
