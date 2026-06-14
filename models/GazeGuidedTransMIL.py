import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim//8,
            heads=8,
            num_landmarks=dim//2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class GazeGuidedTransMIL(nn.Module):
    def __init__(self, n_classes, device,input_dim=1024, gaze_csv_path=None, lambda_attention=0.1):
        super(GazeGuidedTransMIL, self).__init__()
        
        self.lambda_attention = lambda_attention
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.attention_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.pos_layer = PPEG(dim=512)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, n_classes)
        self.device = device
        self.n_classes = n_classes
        
        self.gaze_data = None
        if gaze_csv_path:
            self.load_gaze_data(gaze_csv_path)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def load_gaze_data(self, gaze_csv_path):
        try:
            print("Loading gaze data from CSV...")
            self.gaze_df = pd.read_csv(gaze_csv_path)
            print(f"Loaded gaze data for {self.gaze_df['IMAGE'].nunique()} images")
            
            self.gaze_data_by_image = {}
            for image_name, group in self.gaze_df.groupby('IMAGE'):
                gaze_points = []
                for _, row in group.iterrows():
                    x = row['CURRENT_FIX_X'] 
                    y = row['CURRENT_FIX_Y']
                    duration = row['CURRENT_FIX_DURATION']
                    gaze_points.append((x, y, duration))
                self.gaze_data_by_image[image_name] = np.array(gaze_points)
        except Exception as e:
            print(f"Error loading gaze data: {e}")
            self.gaze_data_by_image = {}

    def get_gaze_data(self, image_name):
        if not hasattr(self, 'gaze_data_by_image') or not self.gaze_data_by_image:
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

    def compute_attention_loss(self, pred_attention, image_name, patch_coords):
        try:
            if not hasattr(self, 'gaze_data_by_image') or image_name is None or patch_coords is None:
                return torch.tensor(0.0).to(self.device)
        
            gaze_points = self.get_gaze_data(image_name)
            if len(gaze_points) == 0:
                return torch.tensor(0.0).to(self.device)
        
            if isinstance(patch_coords, torch.Tensor):
                patch_coords_np = patch_coords.cpu().numpy()
            else:
                patch_coords_np = np.array(patch_coords)
        
            true_attention = self.compute_true_attention_weights(gaze_points, patch_coords_np)
            true_attention = torch.tensor(true_attention, dtype=torch.float32).to(self.device)
        
            n_pred = pred_attention.shape[0]
            n_true = true_attention.shape[0]
        
            if n_pred != n_true:
                min_size = min(n_pred, n_true)
                pred_attention = pred_attention[:min_size]
                true_attention = true_attention[:min_size]
            pred_attention = pred_attention.float() 
            true_attention = true_attention.float()
            epsilon = 1e-8
            pred_probs = F.softmax(pred_attention, dim=0)
            gaze_probs = F.softmax(true_attention, dim=0)
            pred_probs = torch.clamp(pred_probs, min=epsilon, max=1.0-epsilon)
            gaze_probs = torch.clamp(gaze_probs, min=epsilon, max=1.0-epsilon)
            pred_probs = pred_probs / pred_probs.sum()
            gaze_probs = gaze_probs / gaze_probs.sum()
            attention_loss = F.kl_div(
                torch.log(pred_probs), 
                gaze_probs, 
                reduction='batchmean'
            )
            attention_loss = torch.clamp(attention_loss, min=0.0)
        
            if hasattr(self, 'debug_count'):
                self.debug_count += 1
            return attention_loss
        
        except Exception as e:
            return torch.tensor(0.0).to(self.device)

    def compute_true_attention_weights(self, gaze_points, patch_coords):
        patch_durations = []
        total_duration = 0
        
        for patch_idx, coord in enumerate(patch_coords):
            x_min, y_min, x_max, y_max = coord
            in_patch_mask = (
                (gaze_points[:, 0] >= x_min) & (gaze_points[:, 0] < x_max) &
                (gaze_points[:, 1] >= y_min) & (gaze_points[:, 1] < y_max)
            )
            
            patch_gaze_points = gaze_points[in_patch_mask]
            patch_duration = np.sum(patch_gaze_points[:, 2]) if len(patch_gaze_points) > 0 else 0
            patch_durations.append(patch_duration)
            total_duration += patch_duration
        if total_duration > 0:
            attention_weights = [dur / total_duration for dur in patch_durations]
        else:
            attention_weights = [1.0 / len(patch_coords)] * len(patch_coords)
        
        return attention_weights

    def forward(self, input, image_name=None, patch_coords=None, return_attention=False):
        h = input.float()  # [B, n, 1024]
        
        h = self._fc1(h)  # [B, n, 512]
        attention_scores = self.attention_branch(h)  # [B, n, 1]
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=-1)  # [B, n]
        weighted_features = h * attention_weights.unsqueeze(-1)  # [B, n, 512]
        
        #---->pad
        H = weighted_features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h_padded = torch.cat([weighted_features, weighted_features[:,:add_length,:]], dim=1)

        #---->cls_token
        B = h_padded.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)
        h_padded = torch.cat((cls_tokens, h_padded), dim=1)

        #---->Translayer x1
        h_padded = self.layer1(h_padded)

        #---->PPEG
        h_padded = self.pos_layer(h_padded, _H, _W)
        
        #---->Translayer x2
        h_padded = self.layer2(h_padded)

        #---->cls_token
        h_padded = self.norm(h_padded)[:, 0]

        #---->predict
        logits = self._fc2(h_padded)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        
        if return_attention:
            return logits, Y_prob, Y_hat, attention_weights
        else:
            return logits, Y_prob, Y_hat, 0

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.randn((1, 6000, 1024)).to(device)
    model = GazeGuidedTransMIL(n_classes=2, device=device)
    print("Model structure:")
    print(model)
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(model.eval())
    results_dict = model(input=data)
    print(results_dict)