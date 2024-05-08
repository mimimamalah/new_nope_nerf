import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

def empty_directory(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def load_and_process_depth(file_name, depth_anything_dir, depth_original_dir, other_dir, additional_dir, depth_anything_after_dir, depth_original_after_dir):
    depth_anything_path = os.path.join(depth_anything_dir, file_name)
    depth_original_path = os.path.join(depth_original_dir, file_name)
    other_depth_path = os.path.join(other_dir, file_name)
    additional_depth_path = os.path.join(additional_dir, file_name)
    depth_anything_after_path = os.path.join(depth_anything_after_dir, file_name)
    depth_original_after_path = os.path.join(depth_original_after_dir, file_name)

    if not os.path.exists(depth_anything_path) or not os.path.exists(depth_original_path):
        raise FileNotFoundError(f"One or both of the specified depth files do not exist for {file_name}.")

    # Load the depth tensors
    depth_anything = np.squeeze(np.load(depth_anything_path)['pred'])
    depth_original = np.squeeze(np.load(depth_original_path)['pred'])
    other_depth = np.squeeze(np.load(other_depth_path)['pred']) 
    additional_depth = np.squeeze(np.load(additional_depth_path)['pred'])
    depth_anything_after = np.squeeze(np.load(depth_anything_after_path)['pred'])
    depth_original_after = np.squeeze(np.load(depth_original_after_path)['pred'])

    return depth_anything, depth_original, other_depth, additional_depth, depth_anything_after, depth_original_after

def create_plots(depth_anything, depth_original, base_filename, graph_output_dir, other_depth, additional_depth, depth_anything_after, depth_original_after):

    plot_histogram(depth_anything, depth_original, base_filename, graph_output_dir, other_depth, additional_depth, depth_anything_after, depth_original_after)
    
    plot_scatter(depth_anything, depth_original, base_filename, graph_output_dir)
    plot_difference_map(depth_anything, depth_original, base_filename, graph_output_dir)

    # Line plot for a vertical slice from top to bottom at the middle
    plot_vertical_slice(depth_anything, depth_original, base_filename, graph_output_dir, depth_anything.shape[1] // 2)
    
    # Line plot for the top left corner, say 10x10 pixels
    plot_area_comparison(depth_anything, depth_original, base_filename, 'top_left', graph_output_dir, (0, 10), (0, 10))

    # Line plot for the central part, say 10x10 pixels
    plot_area_comparison(depth_anything, depth_original, base_filename, 'center', graph_output_dir, (depth_anything.shape[0]//2 - 5, depth_anything.shape[0]//2 + 5), (depth_anything.shape[1]//2 - 5, depth_anything.shape[1]//2 + 5))


def plot_histogram(depth_anything, depth_original, base_filename, graph_output_dir, other_depth=None, additional_depth=None, depth_anything_after=None, depth_original_after=None):
    depth_anything_flat = depth_anything.flatten()
    depth_original_flat = depth_original.flatten()
    
    # Plot the first histogram with depth-anything and depth-original
    fig_hist1 = plt.figure()
    plt.hist(depth_anything_flat, bins=50, alpha=0.5, label='Depth Anything')
    plt.hist(depth_original_flat, bins=50, alpha=0.5, label='Depth Original')
    plt.title('Histogram of Depth Values')
    plt.xlabel('Depth value')
    plt.ylabel('Frequency')
    plt.legend()
    save_plot(fig_hist1, f'{base_filename}_histogram.png', graph_output_dir)

    # If other_depth is provided, plot it in a second histogram
    if other_depth is not None:
        fig_hist2 = plt.figure()
        plt.hist(depth_anything_flat, bins=50, alpha=0.5, label='Depth Anything')
        plt.hist(depth_original_flat, bins=50, alpha=0.5, label='Depth Original')
        plt.hist(other_depth.flatten(), bins=50, alpha=0.5, label='Other Depth')
        plt.title('Histogram of Depth Values with Other Depth')
        plt.xlabel('Depth value')
        plt.ylabel('Frequency')
        plt.legend()
        save_plot(fig_hist2, f'{base_filename}_histogram_with_other.png', graph_output_dir)
    
    # If additional_depth is provided, plot it in a third histogram
    if additional_depth is not None:
        fig_hist3 = plt.figure()
        plt.hist(depth_anything_flat, bins=50, alpha=0.5, label='Depth Anything')
        plt.hist(depth_original_flat, bins=50, alpha=0.5, label='Depth Original')
        plt.hist(additional_depth.flatten(), bins=50, alpha=0.5, label='Additional Depth')
        plt.title('Histogram of Depth Values with Additional Depth')
        plt.xlabel('Depth value')
        plt.ylabel('Frequency')
        plt.legend()
        save_plot(fig_hist3, f'{base_filename}_histogram_with_additional.png', graph_output_dir)
        
    if depth_anything_after is not None and depth_original_after is not None:
        fig_hist4 = plt.figure()
        plt.hist(depth_anything_after.flatten(), bins=50, alpha=0.5, label='Depth Anything After')
        plt.hist(depth_original_after.flatten(), bins=50, alpha=0.5, label='Depth Original After')
        plt.title('Histogram of Depth Values After Transformation')
        plt.xlabel('Depth value')
        plt.ylabel('Frequency')
        plt.legend()
        save_plot(fig_hist4, f'{base_filename}_histogram_after_transformation.png', graph_output_dir)

def plot_scatter(depth_anything, depth_original, base_filename, graph_output_dir):
    depth_anything_flat = depth_anything.flatten()
    depth_original_flat = depth_original.flatten()
    
    fig_scatter = plt.figure()
    plt.scatter(depth_anything_flat, depth_original_flat, alpha=0.05)
    plt.title('Correlation between Depth Tensors')
    plt.xlabel('Depth Anything')
    plt.ylabel('Depth Original')
    plt.grid(True)
    save_plot(fig_scatter, f'{base_filename}_scatter.png', graph_output_dir)

def plot_difference_map(depth_anything, depth_original, base_filename, graph_output_dir):
    difference = np.abs(np.squeeze(depth_anything) - np.squeeze(depth_original))
    
    fig_diff = plt.figure()
    plt.imshow(difference, cmap='viridis')
    plt.colorbar()
    plt.title('Difference Map of Depth Tensors')
    save_plot(fig_diff, f'{base_filename}_difference_map.png', graph_output_dir)

def plot_vertical_slice(depth_anything, depth_original, base_filename, graph_output_dir, column_index):
    fig, ax = plt.subplots()

    # Get the number of rows (height of the image)
    num_rows = depth_anything.shape[0]
    
    # Create an array of vertical positions, which are just the row indices
    vertical_positions = np.arange(num_rows)

    # Plot the vertical slice from the middle column of depth-anything and depth-original
    ax.plot(vertical_positions, depth_anything[:, column_index], label='Depth Anything Middle Column')
    ax.plot(vertical_positions, depth_original[:, column_index], label='Depth Original Middle Column')

    ax.set_title('Depth Gradient from Top to Bottom at Middle')
    ax.set_xlabel('Vertical Position (Pixel Index)')
    ax.set_ylabel('Depth Value')
    ax.legend(loc='best')

    save_plot(fig, f'{base_filename}_middle_column_plot.png', graph_output_dir)

def plot_area_comparison(depth_anything, depth_original, base_filename, label, graph_output_dir, y_indices, x_indices):
    fig, ax = plt.subplots()
    ax.plot(depth_anything[y_indices[0]:y_indices[1], x_indices[0]:x_indices[1]].flatten(), label=f'Depth Anything {label}')
    ax.plot(depth_original[y_indices[0]:y_indices[1], x_indices[0]:x_indices[1]].flatten(), label=f'Depth Original {label}')
    ax.set_title(f'Depth Comparison at {label}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Depth Value')
    ax.legend()
    save_plot(fig, f'{base_filename}_{label}_plot.png', graph_output_dir)

def save_plot(figure, file_name, graph_output_dir):
    figure_path = os.path.join(graph_output_dir, file_name)
    figure.savefig(figure_path)
    plt.close(figure)  # Close the figure to free memory
    print(f"Saved {file_name} to {figure_path}")

# Set up paths and directories
depth_anything_dir = "data/CVLab/hubble_output_resized_paper/dpt-to-use"
depth_original_dir = "data/CVLab/hubble_output_resized_paper/dpt-before-inversing"
other_dir = "data/CVLab/hubble_output_resized_paper/dpt-anything"

depth_anything_after_dir = "data/CVLab/hubble_output_resized_paper/dpt"
depth_original_after_dir = "data/CVLab/hubble_output_resized/dpt"

graph_output_dir = "data/CVLab/hubble_output_resized_paper/graphs"
additional_dir = "data/CVLab/hubble_output_resized_paper/transformed" 
empty_directory(graph_output_dir)

if not os.path.exists(graph_output_dir):
    os.makedirs(graph_output_dir)

# List of files to compare
files = ['depth_0000.npz', 'depth_0145.npz']

for file_name in files:
    depth_anything, depth_original, other_depth, additional_depth, depth_anything_after, depth_original_after = load_and_process_depth(file_name, depth_anything_dir, depth_original_dir, other_dir, additional_dir, depth_anything_after_dir, depth_original_after_dir)
    create_plots(depth_anything, depth_original, file_name.split('.')[0], graph_output_dir, other_depth, additional_depth, depth_anything_after, depth_original_after)
