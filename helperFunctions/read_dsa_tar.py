
import torch

modelPATH = r"E:\Teresa_article collab\91702_80R6w\tiff\trial_3\transformation_results\tag_1\model_epochs5.tar"

checkpoint = torch.load(modelPATH)
print(checkpoint)