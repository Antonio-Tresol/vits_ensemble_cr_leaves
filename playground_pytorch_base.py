import matplotlib.pyplot as plt
import torch
import torchvision
from helper_functions import set_seeds
from helper_functions import plot_loss_curves
from torch import nn
from torchvision import transforms
from torchinfo import summary

# Plot the results
from data_modules import CRLeavesDataModule, Sampling
from going_modular.going_modular import engine
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# Data modules
data_module = CRLeavesDataModule(
    root_dir="CRLeaves/",
    batch_size=32,
    test_size=0.3,
    use_index=False,
    indices_dir="Indices/",
    sampling=Sampling.NONE,
    train_transform=pretrained_vit_transforms,
    test_transform=pretrained_vit_transforms
)

data_module.prepare_data()
data_module.create_data_loaders()

# 4. Change the classifier head 
class_names = data_module.classes.keys()

set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

# pretrained_vit # uncomment for model output 

# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit, 
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)


print(pretrained_vit_transforms)

NUM_WORKERS = os.cpu_count()

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


# Train the classifier head of the pretrained ViT feature extractor model
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=data_module.train_dataloader(),
                                      test_dataloader=data_module.test_dataloader(),
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=1,
                                      device=device)


plot_loss_curves(pretrained_vit_results) 
