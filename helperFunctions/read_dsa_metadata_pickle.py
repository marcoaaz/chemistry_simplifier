import pickle
import os 

# folderDest = 'E:\\Teresa_article collab\\original_tiff\\13-Jul-24_test5_withCr_linear'
# fileName = 'DSA_teresaPaper_sessionVariables.pckl'

folderDest = 'E:\\Teresa_article collab\\Jones\\full_dataset_tiff\\14-Aug-24_test4_linear'
fileName = 'DSA_clinopyroxeneMask_sessionVariables.pckl'


fileDest = os.path.join(folderDest, fileName)

f = open(fileDest, 'rb')

obj = pickle.load(f)
f.close()    
[workingDir, matlabFile, destDir, descrip_pred, 
        imgName_pred, outPct, LEARNING_RATE, RHO, 
        epoch_default, alpha_reg, betha_reg, fraction, 
        test_ratio, n_workers, BATCH_SIZE, n_workers_pred, 
        BATCH_SIZE_pred, EPOCHS, ALPHA, BETA, ADD_SPARSITY] = obj   

print(f"Pct out of output image {outPct}")
print(f"learning rate {LEARNING_RATE}")
print(f"Epochs {EPOCHS}")
print(f"Alpha regularisation {ALPHA}")
print(f"Beta regularisation {BETA}")
print(f"Fraction {fraction}")
print(f"Test ratio {test_ratio}")
print(f"Batch size {BATCH_SIZE}")
