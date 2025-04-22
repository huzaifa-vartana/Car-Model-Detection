import os
import time
import copy
import json
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool, Process
from torchvision import models, transforms as T
from torch.utils.data import DataLoader
from torchvision.io import read_image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.imgs = self.make_dataset()

    def _find_classes(self, dir):
        if not os.path.isdir(dir):
            raise FileNotFoundError(f"Directory not found: {dir}")
        classes = sorted([d.name for d in os.scandir(dir) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    # Add basic check for image file extensions if needed
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                         item = (path, class_index)
                         images.append(item)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, class_index = self.imgs[idx]
        try:
            image = read_image(path)
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            # Return a dummy tensor or skip this sample
            return torch.zeros((3, 299, 299)), -1, path # Indicate error with label -1

        # Convert grayscale to RGB if necessary
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        # Handle RGBA images (take first 3 channels)
        elif image.shape[0] == 4:
             image = image[:3, :, :]

        # Ensure image is 3 channels if not already
        if image.shape[0] != 3:
             print(f"Warning: Image {path} has {image.shape[0]} channels. Skipping or attempting conversion.")
             # Handle unexpected channel counts, e.g., return dummy
             return torch.zeros((3, 299, 299)), -1, path


        image = image.float() / 255.0

        # Apply initial transforms if provided (e.g., augmentations)
        if self.transform:
            image = self.transform(image)

        # Apply resizing mandatory for ResNet input
        # Use transforms.Resize (imported as T)
        resize_transform = T.Resize((224, 224), antialias=True) # Standard ResNet50 size
        image = resize_transform(image)

        # Apply normalization mandatory for pretrained ResNet
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize_transform(image)

        return image, class_index, path

def prepare_dataloader(dataset, batch_size, num_workers=4, pin_memory=False): # Added pin_memory arg
    # Filter out error samples
    dataset.imgs = [(p, i) for p, i in dataset.imgs if i != -1] # Assuming __getitem__ returns -1 for errors

    # Custom collate to stack tensors and filter out errors
    def collate_fn(batch):
        # Remove None or error samples
        batch = [item for item in batch if item is not None and item[1] != -1]
        if not batch:
            return None
        images, labels, paths = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, paths

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader

def initialize_resnet_model(num_classes=4):
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes),
    )

    return model

def initialize_optimizer_and_criterion(model):
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def train_resnet_model(
    model, criterion, optimizer, data_loaders, device, num_epochs=3 # Added model and device args, removed model_state_dict
):
    # model = initialize_resnet_model() # Model is now passed in
    # model.load_state_dict(model_state_dict) # Loading state dict happens before calling this function
    model.to(device) # Move model to the target device

    since = time.time()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    metrics = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epochs": [],
        "time": 0.0, # Store total time here
    }

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Use tqdm for progress bar
            data_iter = tqdm(data_loaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}")

            for inputs, labels, _ in data_iter:
                # Skip batch if collate_fn returned None due to errors
                if inputs is None or labels is None:
                    continue

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total_samples += batch_size

                # Update tqdm description
                data_iter.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / batch_size)


            # Avoid division by zero if a phase has no samples
            if total_samples == 0:
                 print(f"Warning: No samples processed in phase {phase} for epoch {epoch + 1}. Skipping metrics calculation.")
                 epoch_loss = 0.0
                 epoch_acc = 0.0
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples # Use double for precision

            metrics[f"{phase}_loss"].append(epoch_loss)
            metrics[f"{phase}_acc"].append(epoch_acc.item()) # Store as float

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if it's the best validation accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}")

        metrics["epochs"].append(epoch + 1)
        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s")


    time_elapsed = time.time() - since
    metrics["time"] = time_elapsed # Store total time

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics # Return the trained model and metrics

def main(args):
    print(f"Using device: {args.device}")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    pin_memory = (device.type == 'cuda') # Enable pin_memory only for GPU

    try:
        # Define basic transforms (can be expanded)
        # Note: Resizing and Normalization are now done inside __getitem__
        data_transforms = {
             'train': T.Compose([
                 # Add augmentations here if desired, e.g.,
                 # T.RandomHorizontalFlip(),
                 # T.RandomRotation(10),
             ]),
             'val': T.Compose([
                 # Validation typically only needs resize and normalize (handled in __getitem__)
             ]),
         }

        print("Loading datasets...")
        # Pass the appropriate transform to the dataset
        train_dataset = CustomDataset(root_dir=args.train_dir, transform=data_transforms['train'])
        val_dataset = CustomDataset(root_dir=args.val_dir, transform=data_transforms['val'])
        print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")

        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print("Error: Empty dataset(s). Check data directories and dataset implementation.")
             return

        print("Preparing dataloaders...")
        train_loader = prepare_dataloader(train_dataset, args.batch_size, args.num_workers, pin_memory)
        val_loader = prepare_dataloader(val_dataset, args.batch_size, args.num_workers, pin_memory)

        print("Initializing model...")
        # Determine number of classes from the dataset
        num_classes = len(train_dataset.classes)
        print(f"Number of classes detected: {num_classes}")
        model = initialize_resnet_model(num_classes=num_classes)

        # Load pre-trained weights if specified (optional)
        if args.load_weights and os.path.exists(args.load_weights):
            print(f"Loading weights from {args.load_weights}")
            try:
                 # Adjust loading based on whether the model was saved directly or as state_dict
                 model.load_state_dict(torch.load(args.load_weights, map_location=device))
            except Exception as e:
                 print(f"Error loading weights: {e}. Starting with pre-trained ResNet weights.")
        elif args.load_weights:
             print(f"Weight file not found: {args.load_weights}. Starting with pre-trained ResNet weights.")
        else:
             print("Starting with pre-trained ResNet weights (no specific file loaded).")


        optimizer, criterion = initialize_optimizer_and_criterion(model)

        data_loaders = {"train": train_loader, "val": val_loader}

        print("Starting training...")
        trained_model, metrics = train_resnet_model(model, criterion, optimizer, data_loaders, device, args.epochs)

        print("Saving model...")
        torch.save(trained_model.state_dict(), args.save_weights)
        print(f"Model weights saved to {args.save_weights}")

        print("Saving metrics...")
        # Convert tensor metrics to float for JSON serialization if needed
        if 'val_acc' in metrics and metrics['val_acc']:
             metrics['best_val_acc'] = max(metrics['val_acc'])
        else:
             metrics['best_val_acc'] = 0.0

        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, list):
                serializable_metrics[k] = [float(item) if isinstance(item, (torch.Tensor, float, int)) else item for item in v]
            elif isinstance(v, (torch.Tensor, float, int)):
                serializable_metrics[k] = float(v)
            else:
                serializable_metrics[k] = v


        with open(args.output_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"Metrics saved to {args.output_file}")


    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a ResNet model for car classification.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training') # Reduced default batch size
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes for data loading') # Reduced default workers
    parser.add_argument('--train_dir', type=str, default='RevisedData/train', help='Path to the training data directory')
    parser.add_argument('--val_dir', type=str, default='RevisedData/validation', help='Path to the validation data directory')
    parser.add_argument('--load_weights', type=str, default=None, help='Path to load pre-trained model weights from (optional)')
    parser.add_argument('--save_weights', type=str, default='resnet_model_trained.pth', help='Path to save the trained model weights')
    parser.add_argument('--output_file', type=str, default='training_metrics.json', help='Path to save the training metrics JSON file')

    args = parser.parse_args()


    print(f"Starting training run with config: {args}")
    main(args)
    print("Training run finished.")
