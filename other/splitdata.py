import os
import shutil
import numpy as np

def split_data_iid(data_dir, num_clients):

    # Get all subdirectories (each subdirectory represents a label)
    labels_dirs = [os.path.join(data_dir, label_dir) for label_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label_dir))]
    
    # Collect all images from each label
    all_images = []

    for label_dir in labels_dirs:
        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]
        all_images.extend(images)
    
    # Shuffle all images to ensure randomness
    np.random.shuffle(all_images)  
    
    # Split images into num_clients parts
    total_images = len(all_images)
    images_per_client = total_images // num_clients
    
    for i in range(num_clients):
        
        client_folder = os.path.join(data_dir, f'client_{i+1}')
        os.makedirs(client_folder, exist_ok=True)

        start_idx = i * images_per_client
        end_idx = start_idx + images_per_client if i < num_clients - 1 else total_images
        client_images = all_images[start_idx:end_idx]
        
        # Copy images to new client directories
        for img_path in client_images:
            label = os.path.basename(os.path.dirname(img_path))

            client_label_dir = os.path.join(client_folder, label)
            os.makedirs(client_label_dir, exist_ok=True)
            shutil.copy2(img_path, client_label_dir)
        

def split_data_noniid(data_dir, num_clients):
    
    labels_dirs = [os.path.join(data_dir, label_dir) for label_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label_dir))]
    
    # Sort labels directories by number of images
    for i, label_dir in enumerate(labels_dirs):

        client_num = i % num_clients
        client_folder = os.path.join(data_dir, f'client_{client_num+1}')
        os.makedirs(client_folder, exist_ok=True)
        
        images = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]
        client_label_dir = os.path.join(client_folder, os.path.basename(label_dir))
        os.makedirs(client_label_dir, exist_ok=True)
        
        # Copy images to new client directories
        for img_path in images:
            shutil.copy2(img_path, client_label_dir)



if __name__ == '__main__':
    data_dir = './data/train'
    num_clients = 2

    #split_data_iid(data_dir, num_clients)
    split_data_noniid(data_dir, num_clients)