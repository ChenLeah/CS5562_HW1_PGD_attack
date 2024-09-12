import argparse
import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from load_data import load_results

'''
For evaluating finetune model performance.
'''

parser = argparse.ArgumentParser(description="Testing a Resnet50 fintuned model")
parser.add_argument('--model', type=str, help='the name of model to evaluate', default="version2_4000_mix")
parser.add_argument('--modeldir', type=str, help='the name of the folder of finetune model', default="finetune_models")
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_PATH = os.path.join(args.modeldir, MODEL_NAME)

# Step 1: Load findtune model
print(f'Loading finetune model {MODEL_NAME} ...')

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Initialize the model
model.load_state_dict(torch.load(MODEL_PATH))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Step 2: Load hidden test data
print('Loading hidden test data...')

#TODO: changed here for different test dataset
path='results/esp_3_b5_bs100_sd849'
_,adv_acc, adv_images,labels=load_results(path)

# Move the test data to the appropriate device
adv_images = adv_images.to(device)
labels = labels.to(device)


# Step 3: Evaluation 
with torch.no_grad():
    outputs = model(adv_images).softmax(1)  # Forward pass
    predicted = outputs.argmax(dim=1).cpu()
    
    # Calculate accuracy
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    accuracy = 100 * correct / total

print(f"Finetune Test Accuracy: {accuracy:.2f}%")
print(f"Original Test Accuracy: {adv_acc*100:.2f}%")