# train.py

import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from PIL import ImageFile
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# 从 model.py 文件中导入 resnet50 模型构造函数
from model import resnet50


def matplot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_path, 'Loss.pdf'), dpi=500, format='pdf')
    plt.savefig(os.path.join(save_path, 'Loss.svg'), dpi=500, bbox_inches='tight')
    plt.close()


def matplot_acc(train_acc, val_acc, save_path):
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_path, 'Accuracy.pdf'), dpi=500, format='pdf')
    plt.savefig(os.path.join(save_path, 'Accuracy.svg'), dpi=500, bbox_inches='tight')
    plt.close()


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize])}

    train_data_path = r'/tmp/pycharm_project_845/data/ISIC2018/train'
    val_data_path = r'/tmp/pycharm_project_845/data/ISIC2018/val'
    train_dataset = datasets.ImageFolder(train_data_path, transform=data_transform["train"])
    train_num = len(train_dataset)

    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    num_classes = len(cla_dict)

    output_folder = r'/root/autodl-tmp/mlp_xiugai/xiaorong/isic2018/DASC'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(output_folder, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(val_data_path, transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    net = resnet50(
        num_classes=num_classes,
        use_dmsm=True,
        use_dasc=True,
        use_dkam=False
    )

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 400
    val_best_acc = 0.0
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    loss_train, acc_train, loss_val, acc_val = [], [], [], []

    # ==================== 新增代码：加载权重并继续训练 ====================
    # 1. 设置是否要加载权重
    resume_from_checkpoint = True # 设置为 True 来加载权重，设置为 False 则从头训练

    # 2. 指定要加载的权重文件路径
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  请在这里修改为您要加载的权重文件的实际路径！                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    checkpoint_path = r'/root/autodl-tmp/mlp_xiugai/xiaorong/isic2018/DASC/save_model/epoch_weights/84.6872.63epoch_193.pth'

    start_epoch = 0  # 默认为0，即从第一个epoch开始

    if resume_from_checkpoint:
        if os.path.exists(checkpoint_path):
            print(f"Info: Loading weights from checkpoint: {checkpoint_path}")
            # 加载权重文件
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))

            # 尝试从文件名解析 epoch 数，以便接着训练
            try:
                # 从 "epoch_122.pth" 中提取 "122"
                last_epoch = int(os.path.basename(checkpoint_path).split('_')[1].split('.')[0])
                start_epoch = last_epoch  # 下一个epoch就是 last_epoch
                print(f"Info: Resuming training from epoch {start_epoch + 1}")
            except (ValueError, IndexError):
                print(f"Warning: Could not parse epoch number from filename. Starting from epoch 1.")
                start_epoch = 0  # 如果文件名格式不对，就从头开始计数
        else:
            print(f"Warning: Checkpoint file not found at '{checkpoint_path}'. Starting training from scratch.")
    else:
        print("Info: `resume_from_checkpoint` is False. Starting training from scratch.")
    # ======================================================================

    model_folder = os.path.join(output_folder, 'save_model')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    epoch_weights_folder = os.path.join(model_folder, 'epoch_weights')
    if not os.path.exists(epoch_weights_folder):
        os.makedirs(epoch_weights_folder)

    # 修改训练循环的起始点
    for epoch in range(start_epoch, epochs):
        net.train()
        running_loss, train_acc_sum = 0.0, 0.0
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{epochs}")
        for data in train_bar:
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            train_acc_sum += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.set_postfix(loss=loss.item())
        scheduler.step()

        train_loss = running_loss / train_steps
        train_accurate = train_acc_sum / train_num

        net.eval()
        val_acc_sum, valdata_loss = 0.0, 0.0
        predlist, targetlist = [], []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, desc=f"Valid Epoch {epoch + 1}/{epochs}")
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                valloss = loss_function(outputs, val_labels.to(device))
                valdata_loss += valloss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc_sum += torch.eq(predict_y, val_labels.to(device)).sum().item()
                predlist.extend(predict_y.cpu().numpy())
                targetlist.extend(val_labels.cpu().numpy())

        val_loss = valdata_loss / val_steps
        val_accurate = val_acc_sum / val_num

        loss_train.append(train_loss)
        acc_train.append(train_accurate)
        loss_val.append(val_loss)
        acc_val.append(val_accurate)

        print(
            f'[Epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_acc: {train_accurate:.3f} | val_loss: {val_loss:.3f}, val_acc: {val_accurate:.3f}')

        # 保存每个epoch的权重（保持您原有的逻辑）
        epoch_save_path = os.path.join(epoch_weights_folder, f"epoch_{epoch + 1}.pth")
        torch.save(net.state_dict(), epoch_save_path)

        # 保留原有的逻辑：如果当前模型是最佳的，则额外保存为 best_model.pth
        # 注意：这里的 val_best_acc 没有被恢复，它会从 0.0 重新开始寻找最佳模型。
        # 如果需要精确恢复，则需要更复杂的保存/加载机制。但根据您的要求，此处保持简单。
        if val_accurate > val_best_acc:
            val_best_acc = val_accurate
            best_model_save_path = os.path.join(model_folder, 'best_model.pth')
            torch.save(net.state_dict(), best_model_save_path)
            print(f"New best model saved with accuracy: {val_best_acc:.4f} at epoch {epoch + 1}")

            # 仅在找到更优模型时更新详细指标文件
            labels_range = list(range(num_classes))
            acc = accuracy_score(targetlist, predlist)
            F1 = f1_score(targetlist, predlist, average='macro', zero_division=0)
            precision = precision_score(targetlist, predlist, average='macro', zero_division=0)
            recall = recall_score(targetlist, predlist, average='macro', zero_division=0)
            try:
                val_lab = label_binarize(targetlist, classes=labels_range)
                val_pre = label_binarize(predlist, classes=labels_range)
                auc = roc_auc_score(val_lab, val_pre, average='macro', multi_class='ovr')
            except ValueError:
                auc = -1

            with open(os.path.join(output_folder, 'best_metrics.txt'), 'w') as f:
                f.write(f'Best Validation Accuracy: {val_best_acc:.4f} (Epoch {epoch + 1})\n')
                f.write(f'F1-score (macro): {F1:.4f}\n')
                f.write(f'Precision (macro): {precision:.4f}\n')
                f.write(f'Recall (macro): {recall:.4f}\n')
                f.write(f'AUC (macro): {auc:.4f}\n')

    matplot_loss(loss_train, loss_val, output_folder)
    matplot_acc(acc_train, acc_val, output_folder)
    print('Finished Training')
    print(f'Best validation accuracy: {val_best_acc:.4f}')


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()