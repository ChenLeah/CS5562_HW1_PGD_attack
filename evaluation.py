import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from load_data import load_results 

'''
For Question 3 graph.
'''

# Function to plot epsilon vs accuracy
def plot_epsilon_vs_accuracy(epsilon_values, accuracies):
    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Epsilon (eps)')
    plt.ylabel('Accuracy on Adversarial Images')
    plt.title('Epsilon vs Accuracy on Adversarial Images (batch_num 10,batch_size 100, alpha 2/255)')
    plt.grid(True)
    plt.show()

# Function to visualize top 5 adversarial images
# def tensor_to_image(tensor):
#     denormalize = transforms.Normalize(
#         mean=[-0.485, -0.456, -0.406],
#         std=[1/0.229, 1/0.224, 1/0.225]
#     )
#     img = denormalize(tensor)
#     img = img.permute(1, 2, 0).cpu().numpy()
#     img = img.clip(0, 1)
#     return img

# def visualize_adv_images(adv_images, num_images_to_show=10):
#     plt.figure(figsize=(12, 8))
#     for i in range(num_images_to_show):
#         plt.subplot(1, num_images_to_show, i+1)
#         img = tensor_to_image(adv_images[i])
#         plt.imshow(img)
#         plt.title(f"Adversarial Image {i+1}")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# Main function to gather results, plot accuracy vs epsilon, and show adversarial images
def main(results_paths, epsilon_values):
    accuracies = []
    
    for path in results_paths:
        _, adv_acc, adv_images, _ = load_results(path)
        accuracies.append(adv_acc)
        #visualize_adv_images(adv_images)

    # Plotting epsilon vs accuracy
    plot_epsilon_vs_accuracy(epsilon_values, accuracies)
    
    # Visualizing top 5 adversarial images from the first set of results
    

# Example usage:
# Define paths to your saved result files
results_paths = ['results/esp_1_b1_bs100','results/esp_3_b1_bs100','results/esp_5_b1_bs100','results/esp_10_b1_bs100','results/esp_20_b1_bs100','results/esp_30_b1_bs100','results/esp_40_b1_bs100']
epsilon_values = [0.01, 0.03, 0.05, 0.1,0.2, 0.3, 0.4]

# Run the main function to process the results and display
main(results_paths, epsilon_values)
