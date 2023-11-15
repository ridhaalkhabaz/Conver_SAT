from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.setrecursionlimit(10**6)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".PNG") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            images.append(np.array(img))
    return images

def visualize_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

def numIslands(grid):
    rows, cols = len(grid), len(grid[0])
    num_islands = 0

    def dfs(r, c):
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == 0:
            return
        grid[r][c] = 0
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                num_islands += 1
                dfs(r, c)

    return num_islands

def create_labels():
    folder = '/Users/hari/Downloads/labelsformldsproject' #change path
    images = load_images_from_folder(folder)
    image_labels = []

    for img in images:
        image_labels.append(numIslands(img))

    num_classes = max(image_labels)
    
    return num_classes, image_labels



