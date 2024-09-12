import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights
from load_data import load_results
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Fine-tuning a Resnet50 model")
parser.add_argument('--results', type=str, help='name of the file to save the results to', required=True)
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='finetune_models')
args = parser.parse_args()

RESULTS_DIR = args.resultsdir
RESULTS_PATH = os.path.join(RESULTS_DIR, args.results)

# pre-fintuede model path
MODEL_PATH = None
'''
Accuracy of each fintune model ():
test case: esp3_b5_bs100_sd849 original accuracy: 3%
model name| input pretrained model| dataset| accuracy
1. 'version1_3000_mix' |  origianl model     | 'results/esp_1_b5_bs100_sd678','results/esp_2_b10_bs100','results/esp_5_b10_bs100_sd32','results/esp_10_b3_bs100_sd129','results/esp_30_b2_bs100_sd571'| 74%
2. 'version1_3500_mix' | 'version1_3000_mix' | 'results/esp_50_b5_bs100_sd29'   | 70%
3. 'version1_4000_mix' | 'version1_3000_mix' | 'results/esp_3_b10_bs100_sd3579' | 71%
4. 'version1_4000_mix1'| 'version1_3000_mix' | 'results/esp_8_b10_bs100_sd9'    | 76%
5. 'version2_4000_mix' |  origianl model     | dataset of 1+4 | 79.4 %
'''
ADVIMAEG_PATHS = ['results/esp_8_b10_bs100_sd9','results/esp_1_b5_bs100_sd678','results/esp_2_b10_bs100','results/esp_5_b10_bs100_sd32','results/esp_10_b3_bs100_sd129','results/esp_30_b2_bs100_sd571']

# Step 1: Initialize model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 2: Load and preprocess data
print("Loading adversarial images...")

# the adversarial images and labels loaded in torch tensors
all_adv_images = []
all_labels =[]
for path in ADVIMAEG_PATHS:
    _, _, adv_img, labels = load_results(path)
    all_adv_images.append(adv_img)
    all_labels.append(labels)


# Create a DataLoader with adversarial images and labels
batch_size = 100
adv_dataset = TensorDataset(torch.cat(all_adv_images, dim=0), torch.cat(all_labels, dim=0))
adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)

# Step 3: Fine-tuning
print("Fine-tuning model on adversarial images...")

model.train()
epochs = 5  # mostly reach 99% at 3th epoch


for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"Epoch [{epoch+1}/{epochs}]")

    # Wrap the loader in tqdm to show progress
    for batch_idx, (inputs, labels) in enumerate(tqdm(adv_loader, desc=f"Training Epoch {epoch+1}")):
        # Move data to the appropriate device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
        
        # Update correct predictions
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Print epoch stats
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(adv_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")


print("Fine-tuning completed.")

torch.save(model.state_dict(), RESULTS_PATH)