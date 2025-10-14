# ============================
# garbage overflow and littering - ResNet50 Fine-tuning (Kaggle-ready)
# With Horizontal Augmentation (before splitting) - IMPROVED VERSION
# ============================

import os
import cv2
import shutil
import random
import math
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

# Optional: for metrics
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Step 1: Paths & Config
# ----------------------------
DATASET_DIR   = "/kaggle/input/garbage-with-waterlogging"  # <-- CHANGE if needed
FRAME_DIR     = "processed_dataset"
DATA_SPLIT_DIR= "split_dataset"

N_FRAMES      = 30
SEED          = 42
BATCH_SIZE    = 64
EPOCHS        = 100
BASE_LR       = 3e-4
MIN_LR        = 1e-6
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
NUM_WORKERS   = 4

CLASSES       = ["negative", "positive"]  # Matches your folder structure

# Seeds
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ----------------------------
# Step 2: Frame Extraction (IMPROVED)
# ----------------------------
def ensure_clean_dir(p):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, save_dir, n_frames=N_FRAMES):
    """Extract frames from video with better error handling"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: No frames found in video {video_path}")
        cap.release()
        return
    
    # Better frame sampling - ensure we get frames from throughout the video
    frame_indices = set(range(0, total_frames, max(1, total_frames // n_frames)))
    if len(frame_indices) > n_frames:
        frame_indices = sorted(random.sample(list(frame_indices), n_frames))
    
    frame_count, idx = 0, 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            out_path = Path(save_dir) / f"{video_path.stem}_frame{extracted_count:04d}.jpg"
            # Check if frame is valid before saving
            if frame is not None and frame.size > 0:
                success = cv2.imwrite(str(out_path), frame)
                if success:
                    extracted_count += 1
                else:
                    print(f"Warning: Failed to save frame {out_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path.name}")

print("Extracting frames from videos and copying images...")

# Ensure directories exist
ensure_clean_dir(FRAME_DIR)

for label in ["positive", "negative"]:
    src_dir = Path(DATASET_DIR) / label
    dst_dir = Path(FRAME_DIR) / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if not src_dir.exists():
        print(f"Warning: Source directory {src_dir} does not exist")
        continue

    # Process videos (expanded extensions)
    video_exts = ["*.avi", "*.mp4", "*.mov", "*.mkv", "*.wmv", "*.flv"]
    vids = []
    for ext in video_exts:
        vids.extend(list(src_dir.glob(ext)))
        vids.extend(list(src_dir.glob(ext.upper())))  # Include uppercase extensions
    
    print(f"Found {len(vids)} videos in {label} category")
    for f in tqdm(vids, desc=f"Videos -> {label}"):
        extract_frames(f, dst_dir, n_frames=N_FRAMES)

    # Process images (expanded extensions)
    img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    imgs = []
    for ext in img_exts:
        imgs.extend(list(src_dir.glob(ext)))
        imgs.extend(list(src_dir.glob(ext.upper())))  # Include uppercase extensions
    
    print(f"Found {len(imgs)} images in {label} category")
    for f in tqdm(imgs, desc=f"Images -> {label}"):
        try:
            # Verify image can be read before copying
            test_img = cv2.imread(str(f))
            if test_img is not None:
                shutil.copy(f, dst_dir / f.name)
            else:
                print(f"Warning: Could not read image {f}")
        except Exception as e:
            print(f"Error copying {f}: {e}")

print("Frame extraction + image copy complete.")

# ----------------------------
# Step 2.5: Horizontal Augmentation (IMPROVED)
# ----------------------------
def augment_horizontal(src_dir):
    """Apply horizontal flip augmentation with error handling"""
    for label in ["positive", "negative"]:
        label_dir = Path(src_dir) / label
        if not label_dir.exists():
            print(f"Warning: Directory {label_dir} does not exist for augmentation")
            continue
            
        img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        imgs = []
        for ext in img_exts:
            imgs.extend(list(label_dir.glob(ext)))

        print(f"Augmenting {len(imgs)} images in {label} category")
        successful_augs = 0
        
        for f in tqdm(imgs, desc=f"Augmenting {label}"):
            try:
                img = cv2.imread(str(f))
                if img is None:
                    print(f"Warning: Could not read image {f}")
                    continue
                
                flipped = cv2.flip(img, 1)  # horizontal flip
                out_path = f.with_name(f.stem + "_flip" + f.suffix)
                
                success = cv2.imwrite(str(out_path), flipped)
                if success:
                    successful_augs += 1
                else:
                    print(f"Warning: Failed to save augmented image {out_path}")
                    
            except Exception as e:
                print(f"Error augmenting {f}: {e}")
        
        print(f"Successfully augmented {successful_augs} images in {label} category")

augment_horizontal(FRAME_DIR)
print("Horizontal augmentation complete.")

# ----------------------------
# Step 3: Train/Val/Test Split (IMPROVED)
# ----------------------------
def split_dataset(src_dir, dst_root, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Split dataset with better ratio handling and validation"""
    # Normalize ratios to sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    print(f"Using ratios - Train: {train_ratio:.3f}, Val: {val_ratio:.3f}, Test: {test_ratio:.3f}")

    for label in ["positive", "negative"]:
        src_label_dir = Path(src_dir) / label
        if not src_label_dir.exists():
            print(f"Warning: {src_label_dir} does not exist")
            continue
            
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]:
            files.extend(list(src_label_dir.glob(ext)))

        if len(files) == 0:
            print(f"Warning: No files found in {src_label_dir}")
            continue
            
        print(f"Found {len(files)} files in {label} category")
        
        random.shuffle(files)
        n = len(files)
        n_train = int(train_ratio * n)
        n_val   = int(val_ratio * n)
        
        train_files = files[:n_train]
        val_files   = files[n_train:n_train+n_val]
        test_files  = files[n_train+n_val:]
        
        print(f"{label}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = Path(dst_root) / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for f in split_files:
                try:
                    shutil.copy(f, split_dir / f.name)
                except Exception as e:
                    print(f"Error copying {f}: {e}")

ensure_clean_dir(DATA_SPLIT_DIR)
split_dataset(FRAME_DIR, DATA_SPLIT_DIR)
print("Dataset split complete.")

# ----------------------------
# Step 4: Data Transforms & Loaders (IMPROVED)
# ----------------------------
train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # Additional augmentation
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

try:
    train_data = datasets.ImageFolder(f"{DATA_SPLIT_DIR}/train", transform=train_tfms)
    val_data   = datasets.ImageFolder(f"{DATA_SPLIT_DIR}/val",   transform=val_tfms)
    test_data  = datasets.ImageFolder(f"{DATA_SPLIT_DIR}/test",  transform=val_tfms)
    
    # Check if we have both classes
    if len(train_data.classes) < 2:
        raise ValueError(f"Expected 2 classes, found {len(train_data.classes)}: {train_data.classes}")
    
    print(f"Classes found: {train_data.classes}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Check class distribution
    train_targets = torch.tensor([train_data.targets])
    unique, counts = torch.unique(train_targets, return_counts=True)
    print("Training class distribution:")
    for i, (cls, count) in enumerate(zip(train_data.classes, counts)):
        print(f"  {cls}: {count} samples ({count/len(train_data)*100:.1f}%)")

except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, pin_memory=True)

# ----------------------------
# Step 5: Model (ResNet50 Fine-tuning) - IMPROVED
# ----------------------------
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Fine-tune all layers but with different learning rates
for p in model.parameters():
    p.requires_grad = True

# Replace final layer
num_classes = len(train_data.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# Class weights for imbalanced datasets
class_counts = torch.bincount(torch.tensor(train_data.targets))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Different learning rates for different parts of the model
backbone_params = []
classifier_params = []

for name, param in model.named_parameters():
    if 'fc' in name:
        classifier_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': BASE_LR * 0.1},  # Lower LR for pretrained layers
    {'params': classifier_params, 'lr': BASE_LR}       # Higher LR for new classifier
], weight_decay=WEIGHT_DECAY)

# Improved scheduler
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(0.05 * total_steps)  # Reduced warmup

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(MIN_LR / BASE_LR, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ----------------------------
# Step 6: Training Loop (IMPROVED)
# ----------------------------
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_targets.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    
    return loss_sum / max(1, total), correct / max(1, total), all_targets, all_preds

def train(epochs=EPOCHS):
    best_val_acc, best_epoch, no_improve = 0.0, 0, 0
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            if batch_idx % 10 == 0:  # Update progress less frequently
                pbar.set_postfix({
                    "loss": f"{running_loss / max(1, total):.4f}",
                    "acc":  f"{correct / max(1, total):.4f}",
                    "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # Validation
        val_loss, val_acc, _, _ = evaluate(val_loader)
        
        train_losses.append(running_loss / total)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {running_loss / total:.4f} | Train Acc: {correct / total:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs,
            }, "best_model.pth")
            print(f"✅ New best model saved! Val Acc: {best_val_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1

        # Periodic checkpoints
        if (epoch + 1) % 5 == 0:
            ckpt_name = f"model_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, ckpt_name)
            print(f"💾 Checkpoint saved: {ckpt_name}")

        # Early stopping
        if no_improve >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
            break
    
    print(f"Training complete! Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    return train_losses, val_losses, val_accs

# Train the model
train_losses, val_losses, val_accs = train(EPOCHS)

# ----------------------------
# Step 7: Final Test Evaluation (ISOLATED)
# ----------------------------
print("\n" + "="*60)
print("CREATING TEST LOADER FOR FINAL EVALUATION...")
print("="*60)

# NOW create test loader for final evaluation only
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f"Test set: {len(test_data)} samples")

try:
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
except Exception as e:
    print(f"Error loading best model: {e}")
    print("Using current model state for evaluation")

model.eval()
test_loss, test_acc, y_true, y_pred = evaluate(test_loader)

print(f"\n{'='*50}")
print(f"FINAL TEST RESULTS")
print(f"{'='*50}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"{'='*50}")

try:
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=train_data.classes, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate per-class metrics
    for i, class_name in enumerate(train_data.classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name} class:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        
except Exception as e:
    print(f"Error computing detailed metrics: {e}")

# ----------------------------
# Step 8: Video Inference (IMPROVED)
# ----------------------------
def predict_video(video_path, model, n_frames=30, confidence_threshold=0.6):
    """Predict video class with confidence scoring"""
    tmp_dir = Path("temp_frames")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract frames
        extract_frames(Path(video_path), tmp_dir, n_frames=n_frames)
        
        # Get all extracted frames
        frame_files = list(tmp_dir.glob("*.jpg"))
        if len(frame_files) == 0:
            print(f"No frames extracted from {video_path}")
            return None, 0.0
        
        tfms = val_tfms
        probs = []
        
        model.eval()
        with torch.no_grad():
            for f in frame_files:
                try:
                    img = datasets.folder.default_loader(str(f))
                    x = tfms(img).unsqueeze(0).to(device)
                    
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out = model(x)
                        p = torch.softmax(out, dim=1).squeeze(0)
                    
                    probs.append(p.cpu())
                except Exception as e:
                    print(f"Error processing frame {f}: {e}")
                    continue
        
        if len(probs) == 0:
            print("No frames could be processed")
            return None, 0.0
        
        # Calculate mean probabilities
        mean_prob = torch.stack(probs, dim=0).mean(dim=0)
        predicted_class = int(mean_prob.argmax().item())
        confidence = float(mean_prob.max().item())
        
        # Clean up temporary files
        for f in frame_files:
            try:
                os.remove(f)
            except:
                pass
        tmp_dir.rmdir()
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error in video prediction: {e}")
        return None, 0.0

# Example usage:
# prediction, confidence = predict_video("path/to/video.mp4", model)
# if prediction is not None:
#     class_name = train_data.classes[prediction]
#     print(f"Prediction: {class_name} (confidence: {confidence:.3f})")

print("\n🎉 Training and evaluation complete!")
print("Use predict_video() function for inference on new videos.")