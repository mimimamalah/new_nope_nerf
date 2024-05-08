import numpy as np
import matplotlib.pyplot as plt

# Load the depth values from the text file
depth_values = np.loadtxt('depth_values.txt')

# Plot the depth values as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(depth_values, cmap='viridis')  # You can choose any other colormap like 'plasma', 'inferno', etc.
plt.colorbar(label='Depth value')
plt.title('Depth Map Visualization')
plt.xlabel('Pixel X Coordinate')
plt.ylabel('Pixel Y Coordinate')

# Save the figure to a file
plt.savefig('depth_map_visualization.png', dpi=300, bbox_inches='tight')

alpha_values = np.loadtxt('alpha_values.txt')

# Plot the depth values as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(alpha_values, cmap='viridis')  # You can choose any other colormap like 'plasma', 'inferno', etc.
plt.colorbar(label='Alpha value')
plt.title('Alpha Map Visualization')
plt.xlabel('Pixel X Coordinate')
plt.ylabel('Pixel Y Coordinate')

# Save the figure to a file
plt.savefig('alpha_map_visualization.png', dpi=300, bbox_inches='tight')
