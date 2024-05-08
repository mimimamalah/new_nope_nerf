import trimesh
import numpy as np
import os
from functools import reduce

def load_meshes(directory):
    # Load all meshes that include 'inverted' in the filename
    files = [f for f in os.listdir(directory) if 'inverted' in f and f.endswith('.ply')]
    meshes = [trimesh.load(os.path.join(directory, f)) for f in sorted(files, key=lambda x: int(x.split('_')[1]))]
    return meshes

def align_meshes(meshes):
    # Placeholder for alignment, replace this with actual alignment logic
    # This example assumes meshes are approximately aligned and just transforms them closer
    for i in range(1, len(meshes)):
        # Calculate the centroid of mesh i-1
        prev_centroid = meshes[i-1].centroid
        # Translate mesh i to align with mesh i-1 by centroids
        translation = prev_centroid - meshes[i].centroid
        meshes[i].apply_translation(translation)
    return meshes

def merge_meshes(meshes):
    # Combine all meshes into a single mesh
    combined_mesh = reduce(lambda x, y: x + y, meshes)
    return combined_mesh


directory = 'out/CVLab/hubble_output_resized_paper/extraction/extracted_images/bspline/meshes'
meshes = load_meshes(directory)
aligned_meshes = align_meshes(meshes)
final_mesh = merge_meshes(aligned_meshes)

final_mesh.export('final_merged_mesh.ply')
#final_mesh.show()
