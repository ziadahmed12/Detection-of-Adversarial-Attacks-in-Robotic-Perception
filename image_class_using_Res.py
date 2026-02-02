import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import argparse
import os
from PIL import Image
import numpy as np

# Define data transformations for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define the data directory
data_dir = 'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\prediction'

# Create data loaders with only the first 100 classes
all_classes = os.listdir(os.path.join(data_dir, 'train'))  # Get all class names
limited_classes = all_classes[:100]  # Limit to the first 100 classes

# Create a mapping of original class indices to limited class indices
class_to_idx = {class_name: idx for idx, class_name in enumerate(limited_classes)}

# Create a custom dataset that only includes the first 100 classes
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    for x in ['train', 'val']
}

# Update the class indices in the dataset
for dataset in image_datasets.values():
    dataset.class_to_idx = class_to_idx
    dataset.classes = limited_classes  # Update the classes to reflect the limited classes

# Create data loaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

# Update class names to reflect the limited classes
class_names = limited_classes[:100]  # Ensure class_names only has 100 classes

# Model selection will be done inside __main__ after dataset and class list are known

# # Training loop
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet on a small dataset')
    parser.add_argument('--arch', choices=['resnet18', 'resnet50'], default='resnet18', help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    args = parser.parse_args()
    arch = args.arch
    pretrained = args.pretrained

    # Create the selected model and adapt final layer to number of classes
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported arch: {arch}')

    # Ensure final layer matches limited classes
    num_classes = len(limited_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze all layers except the final classification layer
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Define the loss function and optimizer (only trainable params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    # Move the model to the GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training block commented out per user request. To enable training again,
    # remove the leading '#' on the lines below or implement a command-line flag.
    #
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            print(f"Phase: {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            # accumulate predictions/labels for metrics (compute on CPU)
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # collect for metrics
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Compute additional metrics for this phase (precision/recall/F1 macro)
            try:
                preds_np = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
                labels_np = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
            except Exception:
                preds_np = np.array([])
                labels_np = np.array([])

            def compute_macro_metrics(y_true, y_pred):
                if y_true.size == 0:
                    return float('nan'), float('nan'), float('nan')
                labels = np.unique(np.concatenate([y_true, y_pred]))
                precisions = []
                recalls = []
                f1s = []
                for l in labels:
                    tp = int(((y_pred == l) & (y_true == l)).sum())
                    fp = int(((y_pred == l) & (y_true != l)).sum())
                    fn = int(((y_pred != l) & (y_true == l)).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
                    rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                    if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0:
                        f1 = 2 * prec * rec / (prec + rec)
                    else:
                        f1 = float('nan')
                    if not np.isnan(prec):
                        precisions.append(prec)
                    if not np.isnan(rec):
                        recalls.append(rec)
                    if not np.isnan(f1):
                        f1s.append(f1)
                p_macro = float(np.mean(precisions)) if precisions else float('nan')
                r_macro = float(np.mean(recalls)) if recalls else float('nan')
                f_macro = float(np.mean(f1s)) if f1s else float('nan')
                return p_macro, r_macro, f_macro

            p_macro, r_macro, f_macro = compute_macro_metrics(labels_np, preds_np)
            # Print rounded to 1 decimal
            def _fmt1(x):
                try:
                    if x is None or np.isnan(x):
                        return float('nan')
                    return round(float(x), 1)
                except Exception:
                    return float('nan')

            print(f"{phase} Metrics: Precision_macro={_fmt1(p_macro):.1f} Recall_macro={_fmt1(r_macro):.1f} F1_macro={_fmt1(f_macro):.1f}")

            # Append metrics to Excel workbook (one sheet named 'resnet 18 metrics')
            try:
                from openpyxl import Workbook, load_workbook
                out_dir = 'ress'
                os.makedirs(out_dir, exist_ok=True)
                xlsx_path = os.path.join(out_dir, f'{arch}_metrics.xlsx')
                sheet_name = f'{arch} metrics'
                # Prepare header and row (round numeric values to 1 decimal)
                header_row = ['epoch', 'phase', 'loss', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
                row = [
                    epoch + 1,
                    phase,
                    _fmt1(epoch_loss) if not np.isnan(epoch_loss) else None,
                    _fmt1(epoch_acc) if not np.isnan(epoch_acc) else None,
                    _fmt1(p_macro) if not np.isnan(p_macro) else None,
                    _fmt1(r_macro) if not np.isnan(r_macro) else None,
                    _fmt1(f_macro) if not np.isnan(f_macro) else None,
                ]
                if os.path.exists(xlsx_path):
                    wb = load_workbook(xlsx_path)
                    if sheet_name not in wb.sheetnames:
                        ws = wb.create_sheet(sheet_name)
                        ws.append(header_row)
                    else:
                        ws = wb[sheet_name]
                else:
                    wb = Workbook()
                    # remove default sheet
                    if 'Sheet' in wb.sheetnames:
                        std = wb['Sheet']
                        wb.remove(std)
                    ws = wb.create_sheet(sheet_name)
                    ws.append(header_row)
                ws.append(row)
                wb.save(xlsx_path)
            except Exception as e:
                print(f"Could not write Excel metrics (openpyxl missing or error): {e}")

    print("Training complete!")

    # Save the model
    torch.save(model.state_dict(), f'image_classifier_{arch}.pth')


    import torch
    from torchvision import models, transforms
    from PIL import Image

    # Load the saved model for inference (use same arch)
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    else:
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(f'image_classifier_{arch}.pth'))
    model.eval()

    # Create a new smaller model (example: 2-output) and copy the first 2 output units from trained model
    if arch == 'resnet18':
        new_model = models.resnet18(pretrained=pretrained)
    else:
        new_model = models.resnet50(pretrained=pretrained)
    new_model.fc = nn.Linear(new_model.fc.in_features, 2)
    new_model.fc.weight.data = model.fc.weight.data[0:2]
    new_model.fc.bias.data = model.fc.bias.data[0:2]


    # Load and preprocess the unseen image
    import torch
    from torchvision import models, transforms
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    # Load and preprocess the unseen images
    image_paths = [
        



        # 'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0_seg_mask.png',
        # 'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0_seg_mask.png',

        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0.png',
        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.02.png',

        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0.png',
        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.04.png',

        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0.png',
        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.05.png',


        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0.png',
        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.06.png',

        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.0.png',
        'C:\\pHD\\Fast-Gradient-Signed-Method-FGSM\\ress\\seg\\adversarial_eps_0.07.png',


    ]


    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Perform inference and display images
    plt.figure(figsize=(10, 10))  # Adjust figure size to accommodate more images
    # Add a suptitle that shows which architecture was used (ResNet-18 or ResNet-50)
    try:
        arch_display = 'ResNet 18' if arch == 'resnet18' else 'ResNet 50'
    except Exception:
        arch_display = arch
    plt.suptitle(f'Predictions ({arch_display})', fontsize=16)
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

        with torch.no_grad():
            output = model(input_batch)

        _, predicted_class = output.max(1)

        # Debugging output
        print(f"Predicted class index for image {i+1}: {predicted_class.item()}")

        # Map the predicted class to the class name
        try:
            predicted_class_name = class_names[predicted_class.item()]
        except IndexError:
            print(f"Error: Predicted class index {predicted_class.item()} is out of range for class_names.")
            predicted_class_name = "Unknown"  # Fallback in case of error

        print(f'The predicted class for image {i+1} is: {predicted_class_name}')

        # Display the images with the predicted class names
        image = np.array(image)
        plt.subplot(4, 4, i+1)  # Adjust subplot to display images under each other
        plt.imshow(image)
        plt.axis('off')
        plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')

    plt.show()