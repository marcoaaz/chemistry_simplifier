
import os
import shutil

#Multiple version
def make_dir(destDir):
    image_dir = destDir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)           
    
#Keeper version
def mkdir2(destDir_parent):    
    if not os.path.exists(destDir_parent):
        os.mkdir(destDir_parent)
    else:
        basename1 = os.path.basename(destDir_parent)
        print(f"{basename1} folder already existed.")                  

#Remover version
def mkdir1(destDir_parent):    
    if not os.path.exists(destDir_parent):
        os.mkdir(destDir_parent)        
    else:
        basename1 = os.path.basename(destDir_parent)
        print(f"{basename1} folder already existed. Deleting and regenerating.")            
        remove(destDir_parent)
        os.mkdir(destDir_parent)
            
def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
