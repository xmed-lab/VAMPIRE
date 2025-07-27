import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from skimage.draw import line
import os
import pickle
import json
from tqdm import tqdm

def preprocess_mask(mask):
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = binary.shape
    kernel_size = max(3, int(min(h, w) * 0.003)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    opened = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    min_area_threshold = max(30, max_area * 0.02)
    cleaned = np.zeros_like(opened)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_threshold:
            cleaned[labels == i] = 255
    
    return cleaned


def find_endpoints(skeleton):
    skeleton = (skeleton > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
    endpoints = np.argwhere((skeleton == 1) & (neighbor_count == 1))

    return endpoints

def filter_dense_endpoints(endpoints, skeleton, density_threshold=5, search_radius=20):
    endpoint_map = np.zeros_like(skeleton, dtype=np.uint8)
    for row, col in endpoints:
        endpoint_map[row, col] = 255

    filtered_endpoints = []
    for row, col in endpoints:
        row1, row2 = max(0, row - search_radius), min(skeleton.shape[0], row + search_radius + 1)
        col1, col2 = max(0, col - search_radius), min(skeleton.shape[1], col + search_radius + 1)

        local_density = cv2.countNonZero(endpoint_map[row1:row2, col1:col2])

        if local_density < density_threshold:
            filtered_endpoints.append((row, col))

    return np.array(filtered_endpoints)


def trace_skeleton(skeleton, start):
    visited = np.zeros_like(skeleton, dtype=bool)
    path = []
    stack = [start]

    while stack:
        current = stack.pop()
        row, col = current
        if visited[row, col]:
            continue

        visited[row, col] = True
        path.append(current)

        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                new_row, new_col = row + dy, col + dx
                if 0 <= new_row < skeleton.shape[0] and 0 <= new_col < skeleton.shape[1]:
                    if skeleton[new_row, new_col] and not visited[new_row, new_col]:
                        neighbors.append((new_row, new_col))

        neighbors.sort(key=lambda p: (p[0], p[1]))

        for nb in reversed(neighbors):
            stack.append(nb)

    return path



def visualize_and_save(skeleton, endpoints, save_path="skeleton_with_endpoints.png"):
    vis_image = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)

    for row, col in endpoints:
        cv2.circle(vis_image, (col, row), radius=3, color=(255, 0, 0), thickness=-1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(skeleton, cmap="gray")
    ax[0].set_title("Original Skeleton")
    ax[0].axis("off")

    ax[1].imshow(vis_image)
    ax[1].set_title("Skeleton with Endpoints")
    ax[1].axis("off")

    plt.show()

    cv2.imwrite(save_path, vis_image)


def remove_duplicate_paths(paths):
    seen_sets = set()
    filtered_paths = []
    
    for path in paths:
        path_set = frozenset(tuple(p) for p in path)
        if path_set not in seen_sets:
            seen_sets.add(path_set)
            filtered_paths.append(path)
    
    filtered_paths.sort(key=lambda p: (p[0][0], p[0][1]))
    return filtered_paths

def generate_patch_order(paths, image_shape=(448, 448), patch_size=16):
    """
    Generate patch traversal order by interleaving vessel path patches with border patches
    based on spatial proximity, ensuring each patch appears only once.
    Coordinates are in (row, column) format to match numpy array indexing.
    """
    path_patches = []
    visited = set()
    
    for path in paths:
        path_segment = []
        for idx in range(len(path) - 1):
            row1, col1 = path[idx]
            row2, col2 = path[idx + 1]

            rr, cc = line(col1 // patch_size, row1 // patch_size, col2 // patch_size, row2 // patch_size)

            for i, j in zip(rr, cc):
                if (i, j) not in visited:
                    visited.add((i, j))
                    path_segment.append((j, i))
        
        if path_segment:
            path_patches.append(path_segment)
    
    img_h, img_w = image_shape
    patch_h, patch_w = img_h // patch_size, img_w // patch_size
    border_patches = []

    for i in range(patch_h):
        for j in range(patch_w):
            if (i, j) not in visited:
                border_patches.append((i, j))
    
    border_patches.sort(key=lambda x: (x[0], x[1]))
    
    num_path_segments = len(path_patches)
    num_border_patches = len(border_patches)
    patches_per_segment = num_border_patches // (num_path_segments + 1) if num_path_segments > 0 else num_border_patches
    
    final_order = []
    border_idx = 0
    
    def group_border_patches(patches, max_distance=2):
        """Group border patches that are within max_distance of each other."""
        if not patches:
            return []
            
        groups = []
        current_group = [patches[0]]
        
        for patch in patches[1:]:
            is_close = any(
                abs(patch[0] - p[0]) <= max_distance and abs(patch[1] - p[1]) <= max_distance
                for p in current_group
            )
            
            if is_close:
                current_group.append(patch)
            else:
                current_group.sort(key=lambda x: (x[0], x[1]))
                groups.append(current_group)
                current_group = [patch]
        
        if current_group:
            current_group.sort(key=lambda x: (x[0], x[1]))
            groups.append(current_group)
            
        return groups
    
    if patches_per_segment > 0:
        initial_borders = border_patches[border_idx:border_idx + patches_per_segment]
        border_groups = group_border_patches(initial_borders)
        for group in border_groups:
            final_order.extend(group)
        border_idx += patches_per_segment
    
    for i, path_segment in enumerate(path_patches):
        final_order.extend(path_segment)
        
        if i < num_path_segments - 1:
            if patches_per_segment > 0:
                rows = [p[0] for p in path_segment]
                cols = [p[1] for p in path_segment]
                top = min(rows)
                bottom = max(rows)
                left = min(cols)
                right = max(cols)
                
                remaining_borders = border_patches[border_idx:border_idx + patches_per_segment]
                
                remaining_borders.sort(key=lambda x: min(
                    abs(x[0] - top), 
                    abs(x[0] - bottom),
                    abs(x[1] - left), 
                    abs(x[1] - right) 
                ))
                
                border_groups = group_border_patches(remaining_borders)
                
                for group in border_groups:
                    final_order.extend(group)
                
                border_idx += patches_per_segment

    if border_idx < len(border_patches):
        remaining_borders = border_patches[border_idx:]
        border_groups = group_border_patches(remaining_borders)
        for group in border_groups:
            final_order.extend(group)
    
    return final_order

def visualize_orders_on_image(image_path, orders, patch_size=16, save_path='tmp/path.png'):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = image.shape

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    for i in range(len(orders) - 1):
        row1, col1 = orders[i]
        row2, col2 = orders[i + 1]
        
        p1 = (col1 * patch_size + patch_size // 2, row1 * patch_size + patch_size // 2)
        p2 = (col2 * patch_size + patch_size // 2, row2 * patch_size + patch_size // 2)

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)

    start_row, start_col = orders[0]
    end_row, end_col = orders[-1]
    start_center = (start_col * patch_size + patch_size // 2, start_row * patch_size + patch_size // 2)
    end_center = (end_col * patch_size + patch_size // 2, end_row * patch_size + patch_size // 2)

    ax.scatter(*start_center, color='green', s=100, label='Start')
    ax.scatter(*end_center, color='blue', s=100, label='End')

    ax.legend()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.gca().invert_yaxis()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


def get_patch_indice(orders, image_size=448, patch_size=16):
    cols = image_size // patch_size
    patch_indices = [int(y * cols + x) for x, y in orders]
    return patch_indices


if __name__ == "__main__":
    img2order = {}

    for mask_path in tqdm(os.listdir('mask')):
        mask = cv2.imread(os.path.join('mask', mask_path), cv2.IMREAD_GRAYSCALE)
        processed_mask = preprocess_mask(mask)

        skeleton = skeletonize(processed_mask // 255)
        skeleton = (skeleton * 255).astype(np.uint8)

        endpoints = find_endpoints(skeleton)

        filtered_endpoints = filter_dense_endpoints(endpoints, skeleton)
        paths = []
        for i, start in enumerate(filtered_endpoints):
            path = trace_skeleton(skeleton, start)
            if not any(np.array_equal(path, p) for p in paths):
                paths.append(path)
        
        paths = remove_duplicate_paths(paths)
        orders = generate_patch_order(paths)
        patch_indices = get_patch_indice(orders)
        
        img_name = mask_path.split('.')[0][5:]
        img2order[img_name] = patch_indices

    with open("img2order.pkl", "wb") as f:
        pickle.dump(img2order, f)

    img2inverseorder = {}
    for img in img2order:
        order = img2order[img]

        num_patches = len(order)
        inverse_order = [0] * num_patches

        for i, j in enumerate(order):
            inverse_order[j] = i
        
        img2inverseorder[img] = inverse_order
        
    with open("img2inverseorder.pkl", "wb") as f:
        pickle.dump(img2inverseorder, f)
