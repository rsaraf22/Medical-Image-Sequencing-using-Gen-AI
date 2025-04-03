#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


get_ipython().system('pip install torch torchvision transformers')
get_ipython().system('pip install torch torchvision timm')


# In[ ]:



import os
import cv2
import torch
import timm
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import gc
from skimage.metrics import structural_similarity as ssim


# ## Data extraction using Dataloader

# In[ ]:



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images_per_patient=20):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images_per_patient = max_images_per_patient
        self.patients = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_dir = os.path.join(self.root_dir, self.patients[idx])
        image_paths = sorted([os.path.join(patient_dir, f) for f in os.listdir(patient_dir) if os.path.isfile(os.path.join(patient_dir, f))])[:self.max_images_per_patient]

        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (224, 224))
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images_tensor = torch.stack(images)
        return images_tensor, self.patients[idx]

transform = Compose([ToTensor()])

# Instantiate the dataset
dataset = CustomDataset("/scratch/ajalagam/Formatted_Data_PNG", transform=transform) #give your path to the dataset

model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_model.classifier = nn.Identity()  #to get feature vector
vit_model.eval()



# In[ ]:


Validdataset = CustomDataset("/scratch/ajalagam/Valid_Dataset", transform=transform)

# give your path to your  valid_dataset


# ## Feature Extraction 

# In[ ]:


patient_features = {}
for images, patient_id in DataLoader(dataset, batch_size=1):
    images = images.squeeze(0)  
    with torch.no_grad():
      features = vit_model(images).logits
    patient_features[patient_id] = features

# patient_features dictionary contains features for each patient


# In[ ]:


valid_patient_features = {}
for val_images, val_patient_id in DataLoader(Validdataset, batch_size=1):
    val_images = val_images.squeeze(0) 
    with torch.no_grad():
      val_features = vit_model(val_images).logits
    valid_patient_features[val_patient_id] = val_features

# valid_patient_features dictionary contains features for each patient


# In[ ]:


len(patient_features)


# In[ ]:


len(valid_patient_features)


# ## Model Initialization

# In[ ]:


class VAE(nn.Module):
    def __init__(self, feature_dim):
        super(VAE, self).__init__()
        hidden_dim = 512
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(768, feature_dim * 2),
            
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 3 * 224 * 224),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Aggregate features across the batch
        x = x.mean(dim=0, keepdim=True)  # Taking the mean of the batch
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z).view(3, 224, 224), mu, log_var


# In[ ]:


vae = VAE(feature_dim=512).to(device)


# In[ ]:


# Loss function
def vae_loss(reconstructed, target, mu, log_var):
    reconstruction_loss = nn.functional.mse_loss(reconstructed, target)
    kl_div = -0.5* 0.001 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_div

# optimizer
vae_optimizer = optim.Adam(vae.parameters(), lr=1e-3)


# 
# ## Training Loop
# 

# In[ ]:



torch.cuda.empty_cache()
gc.collect()
losses = []

num_epochs = 100
vae.to(device)
for epoch in range(num_epochs):
    total_loss = 0.0
    count = 0
    for patient_id, features in patient_features.items():
        # Split features into input (first 19 images) and target (20th image)
        input_features = features[:-1]
        last_image_index = features.shape[0]
        folder = int(patient_id[0]) # First 19 images
        input_features = input_features.to(device)
        images_tensor, patient_id2 = dataset[count]
        if last_image_index > 20:
            last_image_index = 20
        specific_image_tensor = images_tensor[-1]
        specific_image_tensor = specific_image_tensor.to(device)
        reconstructed, mu, log_var = vae(input_features)
       
    # Compute VAE loss
        loss = vae_loss(reconstructed, specific_image_tensor, mu, log_var)
        torch.cuda.empty_cache()
        gc.collect()

        # Backpropagation and optimization
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()
        total_loss += loss.item()
        count += 1
        t_loss = total_loss / count if count else 0
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {t_loss:.4f}")
    losses.append(t_loss)

# Plot the loss values
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()


print("Training finished.")

torch.save(vae.state_dict(), '/scratch/ajalagam/Savedvae_model.pth') # give path where you want save the model.


# ## Validation Loop

# In[ ]:



torch.cuda.empty_cache()
import gc
gc.collect()
val_losses = []

val_num_epochs = 100
vae.to(device)
for epoch in range(val_num_epochs):
    val_total_loss = 0.0
    count = 0
    for val_patient_id, val_features in valid_patient_features.items():
        val_input_features = val_features[:-1]
        val_last_image_index = val_features.shape[0]
        val_folder = int(val_patient_id[0]) # First 19 images
      
        val_input_features = val_input_features.to(device)
        val_images_tensor, val_patient_id2 = Validdataset[count]
        if val_last_image_index > 20:
            val_last_image_index = 20
       
        val_specific_image_tensor = val_images_tensor[-1]
        val_specific_image_tensor = val_specific_image_tensor.to(device)
        val_reconstructed, val_mu, val_log_var = vae(val_input_features)
        val_loss = vae_loss(val_reconstructed, val_specific_image_tensor, val_mu, val_log_var)
        torch.cuda.empty_cache()
        gc.collect()

        val_total_loss += val_loss.item()
        count += 1
        v_loss = val_total_loss / count if count else 0
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {v_loss:.4f}")
    val_losses.append(v_loss)

# Plot the loss values
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()
plt.show()


print("Validation finished.")


# ## Output Validation and Visualizations

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

key = ('80',)

features_tensor = patient_features[key].to(device)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/80/18.png')  
image2 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/80/19.png') 
image3 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/80/20.png')

image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4)) 

axes[0].imshow(image1_rgb)
axes[0].set_title("Image 1")
axes[0].axis('off') 

axes[1].imshow(image2_rgb)
axes[1].set_title("Image 2")
axes[1].axis('off')  
plt.tight_layout()

plt.show()


# In[ ]:


vae.eval()
with torch.no_grad():
    reconstructed_images, _, _ = vae(features_tensor)
reconstructed_images = reconstructed_images.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_images = (reconstructed_images * 255).astype(np.uint8)


# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4)) 
axes[0].imshow(image3_rgb)
axes[0].set_title("Next Image in Sequence")
axes[0].axis('off')
axes[1].imshow(reconstructed_images)
axes[1].set_title("Generated Next Image in Sequence")
axes[1].axis('off')
plt.tight_layout()

plt.show()


# In[ ]:



gray1 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (224, 224))
gray2 = cv2.cvtColor(reconstructed_images, cv2.COLOR_BGR2GRAY)


similarity, diff = ssim(gray1, gray2, full=True)

plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.show()

print("SSIM:", similarity)


# In[ ]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

key = ('22',)

features_tensor = patient_features[key].to(device)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

# Read the images
image4 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/22/18.png')  
image5 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/22/19.png') 
image6 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/22/20.png')  

image4_rgb = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
image5_rgb = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)
image6_rgb = cv2.cvtColor(image6, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].imshow(image4_rgb)
axes[0].set_title("Image 1")
axes[0].axis('off') 

axes[1].imshow(image5_rgb)
axes[1].set_title("Image 2")
axes[1].axis('off') 

plt.tight_layout()

plt.show()


# In[ ]:


vae.eval()
with torch.no_grad():
    reconstructed_images, _, _ = vae(features_tensor)
reconstructed_images = reconstructed_images.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_images = (reconstructed_images * 255).astype(np.uint8)


# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].imshow(image6_rgb)
axes[0].set_title("Next Image in Sequence")
axes[0].axis('off')  

axes[1].imshow(reconstructed_images)
axes[1].set_title("Generated Next Image in Sequence")
axes[1].axis('off')  
plt.tight_layout()

plt.show()


# In[ ]:




gray1 = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (224, 224))
gray2 = cv2.cvtColor(reconstructed_images, cv2.COLOR_BGR2GRAY)

similarity, diff = ssim(gray1, gray2, full=True)

plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.show()

print("SSIM:", similarity)


# In[ ]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

key = ('30',)

features_tensor = patient_features[key].to(device)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

image7 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/30/18.png')
image8 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/30/19.png') 
image9 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/30/20.png') 
image7_rgb = cv2.cvtColor(image7, cv2.COLOR_BGR2RGB)
image8_rgb = cv2.cvtColor(image8, cv2.COLOR_BGR2RGB)
image9_rgb = cv2.cvtColor(image9, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))  
axes[0].imshow(image7_rgb)
axes[0].set_title("Image 1")
axes[0].axis('off') 

axes[1].imshow(image8_rgb)
axes[1].set_title("Image 2")
axes[1].axis('off') 

plt.tight_layout()

plt.show()


# In[ ]:


vae.eval()
with torch.no_grad():
    reconstructed_images, _, _ = vae(features_tensor)
reconstructed_images = reconstructed_images.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_images = (reconstructed_images * 255).astype(np.uint8)


# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4)) 

axes[0].imshow(image9_rgb)
axes[0].set_title("Next Image in Sequence")
axes[0].axis('off') 

axes[1].imshow(reconstructed_images)
axes[1].set_title("Generated Next Image in Sequence")
axes[1].axis('off') 

plt.tight_layout()

plt.show()


# In[ ]:




gray1 = cv2.cvtColor(image9, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (224, 224))
gray2 = cv2.cvtColor(reconstructed_images, cv2.COLOR_BGR2GRAY)



similarity, diff = ssim(gray1, gray2, full=True)

plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.show()

print("SSIM:", similarity)


# In[ ]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)
key = ('18',)
features_tensor = patient_features[key].to(device)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

image10 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/18/18.png') 
image11 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/18/19.png')
image12 = cv2.imread('/scratch/ajalagam/Formatted_Data_PNG/18/20.png')
image10_rgb = cv2.cvtColor(image10, cv2.COLOR_BGR2RGB)
image11_rgb = cv2.cvtColor(image11, cv2.COLOR_BGR2RGB)
image12_rgb = cv2.cvtColor(image12, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4)) 

axes[0].imshow(image10_rgb)
axes[0].set_title("Image 1")
axes[0].axis('off')  

axes[1].imshow(image11_rgb)
axes[1].set_title("Image 2")
axes[1].axis('off')

plt.tight_layout()

plt.show()


# In[ ]:


vae.eval()
with torch.no_grad():
    reconstructed_images, _, _ = vae(features_tensor)
reconstructed_images = reconstructed_images.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_images = (reconstructed_images * 255).astype(np.uint8)


# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].imshow(image12_rgb)
axes[0].set_title("Next Image in Sequence")
axes[0].axis('off') 

axes[1].imshow(reconstructed_images)
axes[1].set_title("Generated Next Image in Sequence")
axes[1].axis('off') 

plt.tight_layout()

plt.show()


# In[ ]:




gray1 = cv2.cvtColor(image12, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (224, 224))
gray2 = cv2.cvtColor(reconstructed_images, cv2.COLOR_BGR2GRAY)


similarity, diff = ssim(gray1, gray2, full=True)

plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.show()

print("SSIM:", similarity)


# In[ ]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

key = ('37',)

features_tensor = valid_patient_features[key].to(device)


# In[ ]:


import cv2
import matplotlib.pyplot as plt

image13 = cv2.imread('/scratch/ajalagam/Valid_Dataset/37/14.png') 
image14 = cv2.imread('/scratch/ajalagam/Valid_Dataset/37/15.png')  
image15 = cv2.imread('/scratch/ajalagam/Valid_Dataset/37/16.png') 

image13_rgb = cv2.cvtColor(image13, cv2.COLOR_BGR2RGB)
image14_rgb = cv2.cvtColor(image14, cv2.COLOR_BGR2RGB)
image15_rgb = cv2.cvtColor(image15, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].imshow(image13_rgb)
axes[0].set_title("Image 1")
axes[0].axis('off')  

axes[1].imshow(image14_rgb)
axes[1].set_title("Image 2")
axes[1].axis('off') 

plt.tight_layout()

plt.show()


# In[ ]:


vae.eval()
with torch.no_grad():
    reconstructed_images, _, _ = vae(features_tensor)
reconstructed_images = reconstructed_images.squeeze().permute(1, 2, 0).cpu().numpy()
reconstructed_images = (reconstructed_images * 255).astype(np.uint8)


# In[ ]:



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))  

axes[0].imshow(image15_rgb)
axes[0].set_title("Next Image in Sequence")
axes[0].axis('off') 

axes[1].imshow(reconstructed_images)
axes[1].set_title("Generated Next Image in Sequence")
axes[1].axis('off')
plt.tight_layout()

plt.show()


# In[ ]:



gray1 = cv2.cvtColor(image15, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (224, 224))
gray2 = cv2.cvtColor(reconstructed_images, cv2.COLOR_BGR2GRAY)


similarity, diff = ssim(gray1, gray2, full=True)

plt.imshow(diff, cmap='gray')
plt.title("Difference Image")
plt.show()

print("SSIM:", similarity)


# ## Model Summary

# In[ ]:


vit_model.config


# In[ ]:


from torchsummary import summary
summary(vae, input_size=(1, 768))

