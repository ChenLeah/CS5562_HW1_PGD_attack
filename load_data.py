import torch

# Function to load the saved results and gather information
def load_results(results_path):
    checkpoint = torch.load(results_path)
    acc = checkpoint['acc']           # Accuracy on original images
    adv_acc = checkpoint['adv_acc']   # Accuracy on adversarial images
    adv_images = checkpoint['adv_images']  # Adversarial images
    labels = checkpoint['labels']     # Corresponding labels
    return acc, adv_acc, adv_images, labels