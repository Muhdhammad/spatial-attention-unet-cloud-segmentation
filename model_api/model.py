import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pathlib import Path


class CustomDataset(Dataset):

    def __init__(self, red_dir, green_dir, blue_dir, nir_dir, mask_dir, pytorch=True):
        super().__init__()
        self.red_dir = red_dir
        self.green_dir = green_dir
        self.blue_dir = blue_dir
        self.nir_dir = nir_dir
        self.mask_dir = mask_dir

        red_files = [f for f in self.red_dir.iterdir() if f.is_file()]
        self.files = [self.combine_files(f) for f in red_files]
        self.pytorch = pytorch


    def combine_files(self, red_files: Path):
        base_name = red_files.name

        files = {
            'red': red_files,
            'green': self.green_dir / base_name.replace('red', 'green'),
            'blue': self.blue_dir / base_name.replace('red', 'blue'),
            'nir': self.nir_dir / base_name.replace('red', 'nir'),
            'mask': self.mask_dir / base_name.replace('red', 'gt'),
        }

        for key, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f'Missing file: {path} for {red_files}')
        return files


    def __len__(self):
        return len(self.files)


    def open_as_array(self, idx, invert=False, nir_included=False):
        rgb = np.stack([
                np.array(Image.open(self.files[idx]['red'])),
                np.array(Image.open(self.files[idx]['green'])),
                np.array(Image.open(self.files[idx]['blue']))
            ], axis=2)

        if nir_included:
            nir = np.array(Image.open(self.files[idx]['nir']))
            nir = np.expand_dims(nir, 2)
            rgb = np.concatenate([rgb, nir], axis=2)

        if invert:
            rgb = rgb.transpose((2, 0, 1))

        raw_rgb = (rgb / np.iinfo(rgb.dtype).max)
        return raw_rgb

    def open_mask(self,idx, expand_dims=True):
        raw_mask = np.array(Image.open(self.files[idx]['mask']))
        raw_mask = np.where(raw_mask == 255, 1, 0) # Transform the mask into binary array where pixels with value 256(white) become 1(clouds), pixels with 0 or anything else becomes 0(not clouds)

        return np.expand_dims(raw_mask, 0) if expand_dims else raw_mask

    def __getitem__(self, idx):
        X = torch.tensor(self.open_as_array(idx, invert=True, nir_included=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, expand_dims=True), dtype=torch.float32)

        return X, y


class doubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class downSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = doubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class upSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = doubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooling = torch.mean(x, dim=1, keepdim=True)
        max_pooling = torch.max(x, dim=1, keepdim=True)[0] # return on max values and not their indices
        concat = torch.cat([avg_pooling, max_pooling], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        output = x * attention
        return output


class unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_conv1 = downSample(in_channels, 32)
        self.down_conv2 = downSample(32, 64)
        self.down_conv3 = downSample(64, 128)

        self.bottleneck = doubleConv(128, 256)
        self.spatial_attention = SpatialAttention()

        self.up_conv1 = upSample(256, 128)
        self.up_conv2 = upSample(128, 64)
        self.up_conv3 = upSample(64, 32)

        self.out = nn.Conv2d(in_channels=32 , out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        down1, p1 = self.down_conv1(x)
        down2, p2 = self.down_conv2(p1)
        down3, p3 = self.down_conv3(p2)

        b = self.bottleneck(p3)
        b = self.spatial_attention(b)

        up1 = self.up_conv1(b, down3)
        up2 = self.up_conv2(up1, down2)
        up3 = self.up_conv3(up2, down1)

        output = self.out(up3)
        return output

def acc_fn(predb, yb):
    preds = torch.sigmoid(predb)  # Convert logits to probabilities
    preds = (preds > 0.5).float()  # Threshold at 0.5
    return (preds == yb).float().mean()  # Compare with ground truth

def calculate_metrics(y_true, y_pred):
    TP = torch.sum((y_true == 1) & (y_pred == 1)).float()
    TN = torch.sum((y_true == 0) & (y_pred == 0)).float()
    FP = torch.sum((y_true == 0) & (y_pred == 1)).float()
    FN = torch.sum((y_true == 1) & (y_pred == 0)).float()

    jaccard = TP / (TP + FN + FP + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    overall_acc = (TP + TN) / (TP + TN + FP + FN + 1e-10)

    return {
        "Jaccard index": jaccard.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "Specificity": specificity.item(),
        "Overall Accuracy": overall_acc.item()
    }
