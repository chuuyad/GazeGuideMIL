import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


class MyDataset(Dataset):
    def __init__(self, feat_dir, transform=None, split=None, train=True, dset='camelyon',
                 coords_dir=None, gaze_csv_path=None):
        self.train = train
        self.transform = transform
        self.feat_dir = feat_dir
        self.coords_dir = coords_dir
        self.img_id = list(split.keys())
        self.label = list(split.values())
        self.dset = dset
        self.feat_files = self.get_feat_file()
        
        self.gaze_data_by_image = {}
        if gaze_csv_path and os.path.exists(gaze_csv_path):
            self.load_gaze_data(gaze_csv_path)
        
        self.patch_coords_cache = {}
        if self.coords_dir and os.path.exists(self.coords_dir):
            print(f"Loading patch coordinates from {self.coords_dir}")

    def load_gaze_data(self, gaze_csv_path):
        import pandas as pd
        print(f"Loading gaze data from {gaze_csv_path}")
        gaze_df = pd.read_csv(gaze_csv_path)
        
        for image_name, group in gaze_df.groupby('IMAGE'):
            gaze_points = []
            for _, row in group.iterrows():
                x = row['CURRENT_FIX_X']
                y = row['CURRENT_FIX_Y']
                duration = row['CURRENT_FIX_DURATION']
                gaze_points.append((x, y, duration))
            self.gaze_data_by_image[image_name] = np.array(gaze_points)

    def get_gaze_data(self, image_name):
        if not self.gaze_data_by_image:
            return np.array([])
        possible_names = [
            image_name,
            image_name + '.png',
            image_name.replace('.png', ''),
            image_name.split('.')[0]
        ]
        for name in possible_names:
            if name in self.gaze_data_by_image:
                return self.gaze_data_by_image[name]
        return np.array([])

    def get_patch_coords(self, img_name):
        if img_name in self.patch_coords_cache:
            return self.patch_coords_cache[img_name]
        
        patch_coords = None
        if self.coords_dir:
            for ext in ['.npy', '.npz', '.pt', '_coords.npy', '_coords.pt']:
                coord_file = os.path.join(self.coords_dir, f"{img_name}{ext}")
                if os.path.exists(coord_file):
                    if ext.endswith('.pt'):
                        data = torch.load(coord_file, map_location='cpu')
                        if isinstance(data, dict):
                            if 'coords' in data:
                                patch_coords = data['coords']
                            else:
                                patch_coords = list(data.values())[0]
                        else:
                            patch_coords = data
                    else:
                        patch_coords = np.load(coord_file)
                    if patch_coords is not None:
                        break
        
        if patch_coords is None:
            patch_coords = self.infer_patch_coords(img_name)
        
        self.patch_coords_cache[img_name] = patch_coords
        return patch_coords

    def infer_patch_coords(self, img_name):
        feat_files = self.feat_files.get(img_name, [])
        if not feat_files:
            return np.array([])
        data = torch.load(feat_files[0], map_location='cpu')
        if isinstance(data, dict):
            feat = list(data.values())[0]
        else:
            feat = data
        num_patches = feat.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        coords = []
        patch_size = 1.0 / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if len(coords) >= num_patches:
                    break
                x_min = j * patch_size
                y_min = i * patch_size
                x_max = (j + 1) * patch_size
                y_max = (i + 1) * patch_size
                coords.append([x_min, y_min, x_max, y_max])
        return np.array(coords[:num_patches], dtype=np.float32)

    def get_feat_file(self):
        feat_files = {}
        for id in self.img_id:
            if self.dset == 'camelyon':
                feat_files[id] = [os.path.join(self.feat_dir, str(id)+'.pt')]
            else:
                pattern = os.path.join(self.feat_dir, id + '*')
                files = glob.glob(pattern)
                feat_files[id] = sorted(files)  
        return feat_files

    def _load_feature(self, filepath):
        data = torch.load(filepath, map_location='cpu')
        if isinstance(data, dict):
            for key in ['features', 'feats', 'embedding', 'emb']:
                if key in data:
                    return data[key]
            return list(data.values())[0]
        return data

    def __getitem__(self, idx):
        img_name = self.img_id[idx]
        feat_files = self.feat_files[img_name]
        
        feats_list = []
        for fpath in feat_files:
            feat = self._load_feature(fpath)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            feats_list.append(feat)
        
        if len(feats_list) == 1:
            resnet_feats = feats_list[0]
        else:
            resnet_feats = torch.cat(feats_list, dim=0)
        
        target = self.label[idx]
        
        patch_coords = self.get_patch_coords(img_name)
        
        n_feats = resnet_feats.shape[0]
        if patch_coords is not None:
            if patch_coords.shape[0] != n_feats:
                if patch_coords.shape[0] > n_feats:
                    patch_coords = patch_coords[:n_feats]
                else:
                    default_coord = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
                    pad = np.tile(default_coord, (n_feats - patch_coords.shape[0], 1))
                    patch_coords = np.concatenate([patch_coords, pad], axis=0)
        else:
            patch_coords = np.tile([0.0, 0.0, 1.0, 1.0], (n_feats, 1)).astype(np.float32)
        
        patch_coords = np.asarray(patch_coords, dtype=np.float32)

        gaze_data = self.get_gaze_data(img_name)
        
        sample = {
            'img_id': img_name,
            'feat': resnet_feats,
            'target': target,
            'patch_coords': patch_coords,
            'gaze_data': gaze_data
        }
        return sample

    def __len__(self):
        return len(self.img_id)