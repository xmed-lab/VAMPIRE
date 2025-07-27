# finetuning the weights of visionfm while training a multi-class classifier on top

import os
import argparse
import json
import copy
import torch
import torch.backends.cudnn as cudnn
import utils
import models
import numpy as np
from models.head import ClsHead, InfoProjector, InfoProjector, ModelWithPrompt

from pathlib import Path
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,cohen_kappa_score, accuracy_score,precision_score,recall_score
from collections import defaultdict
import csv
import pandas as pd
import cv2
import pickle


import wandb

class OCTAInhouseBinaryDatasetText(Dataset):
    def __init__(self, root, imgSize=512, isTraining=True, transform=None, fold=0, order_pkl_path=None, inverse_order_pkl_path=None, text_path=None):
        super(OCTAInhouseBinaryDatasetText,self).__init__()
        self.root = root
        self.isTraining = isTraining
        self.imgsize = imgSize
        self.fold = fold
        self.allItems = self.getAllPath(root, isTraining)
        
        self.scan_orders = None
        self.inverse_scan_orders = None
        self.text_embeds = None
        
        if order_pkl_path is not None:
            with open(order_pkl_path, "rb") as f:
                self.scan_orders = pickle.load(f)
        if inverse_order_pkl_path is not None:
            with open(inverse_order_pkl_path, "rb") as f:
                self.inverse_scan_orders = pickle.load(f)
        
        if text_path is not None:
            with open(text_path, "r") as f:
                self.text_embeds = json.load(f)
        
        if transform is None:
            if isTraining:
                self.transform= transforms.Compose([
                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.Resize((self.imgsize,self.imgsize)),
                    transforms.RandomCrop(size=self.imgsize, padding=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(10),
                    #transforms.RandomResizedCrop(size=self.imgsize, scale=(0.8, 1.2))
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imgsize,self.imgsize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        else:
            self.transform = transform

    def __getitem__(self, index):
        imageid, highrisk, highglucose, hightc, hightg, hypertension, risk, glucose, tc, tg, systole, diastole, age, gender = self.allItems[index]
        imgS = Image.open(os.path.join(self.root, 'Superficial', imageid)).convert('L')
        imgD = Image.open(os.path.join(self.root, 'Deep', imageid)).convert('L')
        imgC = Image.open(os.path.join(self.root, 'Choriocapillaris', imageid)).convert('L')
        labels = np.array([int(highrisk), int(highglucose), int(hightc), int(hightg), int(hypertension)])
        
        age = int(age)
        gender = np.eye(2)[int(gender)]
        
        imgS = np.asarray(imgS)
        imgD = np.asarray(imgD)
        imgC = np.asarray(imgC)
        img = np.stack((imgS, imgD, imgC), axis=2)
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

        image = self.transform(image)
        
        order, inverse_order = None, None
        if self.scan_orders is not None and self.inverse_scan_orders is not None:
            order = torch.tensor(self.scan_orders[imageid.split('.')[0]], dtype=torch.long)
            inverse_order = torch.tensor(self.inverse_scan_orders[imageid.split('.')[0]], dtype=torch.long)
        
        text_embed = None
        if self.text_embeds is not None:
            # text_embed = self.text_embeds[imageid.split('-')[0]]
            text_embed = self.text_embeds[imageid]

        if order is not None and text_embed is not None:
            return image, age, gender, labels, order, inverse_order, text_embed, imageid
        elif order is not None:
            return image, age, gender, labels, order, inverse_order, imageid
        else:
            return image, age, gender, labels, imageid

    def __len__(self):
        return len(self.allItems)

    def getAllPath(self, root, isTraining):
        items = []
        if isTraining:
            filePath = 'data/Label/folds_info/train_{}.csv'.format(str(self.fold))
        else:
            filePath = 'data/Label/folds_info/test_{}.csv'.format(str(self.fold))

        with open(filePath,'r') as csvFile:
            reader = csv.reader(csvFile)
            header = next(reader)
            for item in reader:
                left_right = item[0].split('_')[0].split('-')[-1]
                front_back = item[0][:-4].split('_')[-1]
                ## select back
                if left_right == '0' and front_back == 'R':         
                    items.append([item[0], item[2], item[3], item[4], item[5], item[6], item[8], item[9], item[10], item[11], item[12], item[13], item[14], item[15]])
                elif left_right == '1' and front_back == 'L':
                    items.append([item[0], item[2], item[3], item[4], item[5], item[6], item[8], item[9], item[10], item[11], item[12], item[13], item[14], item[15]])
        
        return items

    def calculate_weights(self):
        """
        Calculate weights for each sample based on multi-label distribution.
        Returns a list of weights with the same length as the dataset.
        """
        # Get all labels
        all_labels = np.array([[int(x) for x in item[1:6]] for item in self.allItems])  # Extract all labels and convert to int
        num_samples = len(all_labels)
        num_classes = all_labels.shape[1]  # Should be 6 for your case
        
        # Calculate class frequencies
        class_counts = np.sum(all_labels, axis=0)  # Sum for each class
        class_weights = num_samples / (num_classes * class_counts)  # Inverse frequency
        
        # Calculate sample weights
        # For each sample, take the mean of the weights of its positive labels
        sample_weights = []
        for labels in all_labels:
            if np.sum(labels) == 0:  # Handle samples with no positive labels
                weight = np.mean(class_weights)
            else:
                weight = np.mean(class_weights[labels == 1])
            sample_weights.append(weight)
        
        # Normalize weights
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        return sample_weights


def eval_linear(args):
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    
    # fix the seed for reproducibility 
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    transforms.ToTensor(),

    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(args.input_size, args.input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print(f"-------- Current Task: {args.task} Modality: {args.modality} -------")

    dataset_train = OCTAInhouseBinaryDatasetText(root=args.data_path, isTraining=True, transform=train_transform, fold=args.fold, order_pkl_path=args.order_path, inverse_order_pkl_path=args.inverse_order_path, text_path=args.text_path)
    dataset_val = OCTAInhouseBinaryDatasetText(root=args.data_path, isTraining=False, transform=val_transform, fold=args.fold, order_pkl_path=args.order_path, inverse_order_pkl_path=args.inverse_order_path, text_path=args.text_path)

    weights = dataset_train.calculate_weights()
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=args.num_samples, replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = models.__dict__[args.arch](
        img_size = [args.input_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=args.avgpool_patchtokens==1)
    embed_dim = model.embed_dim
    model = model.to(device)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load visionfm pretrained weights
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    linear_classifier = ClsHead(embed_dim=embed_dim*4, num_classes=args.num_labels, layers=3)        
    linear_classifier = linear_classifier.to(device)
     
    info_projector_text = ModelWithPrompt()
    info_projector_text = info_projector_text.to(device)
    
    info_projector = InfoProjector(output_dim=args.info_dim, hidden_dim=args.info_dim//2)  
    info_projector = info_projector.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
        linear_classifier = nn.DataParallel(linear_classifier)
        info_projector = nn.DataParallel(info_projector)
        info_projector_text = nn.DataParallel(info_projector_text)


    optimizer = torch.optim.AdamW(
        [{'params': model.parameters(), 'lr': args.lr * 0.1 * (args.batch_size_per_gpu * utils.get_world_size()) / 256.}, 
            {'params': linear_classifier.parameters()}, 
            {'params': info_projector_text.parameters(), 'lr': args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.}, 
            {'params': info_projector.parameters(), 'lr': args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.}],
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        betas=(0.9, 0.999), weight_decay=0.05
    )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in linear_classifier.parameters()) + sum(p.numel() for p in info_projector.parameters()) + sum(p.numel() for p in info_projector_text.parameters())
    print('number of params:', n_parameters)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_auc": 0.}
    start_epoch = to_restore["epoch"]
    best_auc = to_restore["best_auc"]
    aupr_with_best_auc = 0

    if args.eval:
        test_stats, output, target, test_logs = validate_network_vsl(val_loader, model, info_projector, info_projector_text, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, verbose=True, output_dir=args.output_dir, task='{}_fold{}'.format(args.task, args.fold))

    for epoch in range(start_epoch, args.epochs):
        # train_loader.sampler.set_epoch(epoch)
        model.train()
        linear_classifier.train()
        info_projector.train()
        
        train_stats = train_vsl(args, model, info_projector, info_projector_text, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            info_projector.eval()
            linear_classifier.eval()
            test_stats, output, target, test_logs = validate_network_vsl(val_loader, model, info_projector, info_projector_text, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, verbose=True, output_dir=args.output_dir, task='{}_fold{}'.format(args.task, args.fold))

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'val_{k}': v for k, v in test_stats.items()},
                         **{k: v for k, v in test_logs.items()}}
            
            wandb.log(log_stats)
        
            if utils.is_main_process(): # and (test_stats["auc"] >= best_auc):
                # always only save best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                
                save_dict = {
                    "epoch": epoch + 1,
                    "classifier_state_dict": linear_classifier.state_dict(),
                    "visionfm_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best_finetune.pth"))
                np.save(os.path.join(args.output_dir, 'best.npy'), output)
                np.save(os.path.join(args.output_dir, 'target.npy'), target)

    print("Finetuning of VisionFM completed")

def train_vsl(args, model, info_projector, info_projector_text, linear_classifier, optimizer, loader, epoch, n, avgpool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, age, gender, target, orders, inverse_orders, text_embed, imageids) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True).float()
        gender = gender.to(device, non_blocking=True).float()
        orders = orders.to(device, non_blocking=True)
        inverse_orders = inverse_orders.to(device, non_blocking=True)
        

        age_features, gender_features = info_projector(age, gender)
        text_features = info_projector_text(text_embed, device)
        
        info_features = torch.cat([age_features.unsqueeze(1), gender_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1)
        
        intermediate_output = model(inp, info_features, orders, inverse_orders, device, n)
        output_feat_ = [x[:, 0] for x in intermediate_output]
        output_feat = torch.cat(output_feat_, dim=-1)
        output = linear_classifier(output_feat)
        loss = nn.BCEWithLogitsLoss()(output, target.float())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network_vsl(val_loader, model, info_projector, info_projector_text, linear_classifier, n, avgpool, verbose=False, output_dir=None, task=None):
    model.eval()
    linear_classifier.eval()
    info_projector.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    targets, preds, imageids = [], [], []
    for inp, age, gender, target, orders, inverse_orders, text_embeds, imageid in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True).float()
        gender = gender.to(device, non_blocking=True).float()
        orders = orders.to(device, non_blocking=True)
        inverse_orders = inverse_orders.to(device, non_blocking=True)
        # text_embeds = text_embeds.to(device, non_blocking=True)
        
        age_features, gender_features = info_projector(age, gender)
        text_features = info_projector_text(text_embeds, device)
        
        info_features = torch.cat([age_features.unsqueeze(1), gender_features.unsqueeze(1), text_features.unsqueeze(1)], dim=1)

        # forward
        with torch.no_grad():
            intermediate_output = model(inp, info_features, orders, inverse_orders, device, n)
            output_feat = [x[:, 0] for x in intermediate_output]

            output = torch.cat(output_feat, dim=-1)
        
            if args.adverse:
                output, _ = linear_classifier(output, output_feat, [age_features, gender_features, text_features])
            else:
                output = linear_classifier(output)

            num_class = output.shape[1]
            loss = nn.BCEWithLogitsLoss()(output, target.float())

            output = nn.Sigmoid()(output)

        preds.extend(output.cpu().detach().numpy())
        targets.extend(target.cpu().detach().numpy())
        imageids.extend(imageid)

        metric_logger.update(loss=loss.item())

    targets = np.array(targets)
    preds = np.array(preds)
    predicts = (preds > 0.5).astype(int)

    acc, auc, rec, pre, f1, kappa, aupr = CalMetricMulti(preds, targets, num_class=num_class)
    log_dict = {
            "accuracy": acc,
            "AUC": auc,
            "Recall": rec,
            "Precision": pre,
            "F1": f1,
            "Kappa": kappa,
            "AUPR": aupr
        }

    print('* val loss {losses.global_avg:.4f} '.format(losses=metric_logger.loss))

    if verbose:
        results_path = os.path.join(output_dir, task+'_predicts.xlsx')
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            sheetname = ['highrisk', 'highglucose', 'hightc', 'hightg', 'hypertension', 'hcyabn']
            for sheet_index in range(len(predicts[0])):
                rows = []
                for idx, imageid in enumerate(imageids):
                    rows.append([
                        imageid,
                        predicts[idx][sheet_index],
                        preds[idx][sheet_index],
                        targets[idx][sheet_index]
                    ])

                df = pd.DataFrame(rows, columns=["Image ID", "Score", "Prediction", "Ground Truth"])
                df.to_excel(writer, sheet_name=sheetname[sheet_index], index=False)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets, log_dict


def CalMetricMulti(predictions, targets, num_class):
    epsilon = 1e-8
    # Apply sigmoid to get probabilities
    predictions = predictions#.cpu().numpy()
    targets = targets#.cpu().numpy()
    
    pred_binary = (predictions > 0.3).astype(int)
    
    auc_ = 0
    kappa = 0
    acc = 0
    aupr = 0
    pre, rec, f1 = 0, 0, 0
    total_samples = targets.sum(axis=0)  # number of positive samples for each class

    print("\nPer-class metrics:")
    print("-" * 108)
    print("|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|".format(
        "Class", "Accuracy", "AUC", "Precision", "Recall", "F1", "Kappa", "AUPR"))
    print("-" * 108)
    
    cls_name = ['highRisk', 'highGlucose', 'highTC', 'highTG', 'hypertension']
    for i in range(num_class):
        class_acc = accuracy_score(targets[:, i], pred_binary[:, i])
        class_auc = roc_auc_score(targets[:, i], predictions[:, i])
        class_pre = precision_score(targets[:, i], pred_binary[:, i], zero_division=0)
        class_rec = recall_score(targets[:, i], pred_binary[:, i], zero_division=0)
        class_f1 = f1_score(targets[:, i], pred_binary[:, i], zero_division=0)
        class_kappa = cohen_kappa_score(targets[:, i], pred_binary[:, i])
        
        class_aupr = average_precision_score(targets[:, i], predictions[:, i])
        
        acc += class_acc
        auc_ += class_auc
        kappa += class_kappa
        aupr += class_aupr
        pre += class_pre
        rec += class_rec
        f1 += class_f1
        
        print("|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|".format(
                cls_name[i], class_acc, class_auc, class_pre, class_rec, class_f1, class_kappa, class_aupr))

    print("-" * 108)
    
    auc_ = auc_ / num_class
    kappa = kappa / num_class
    acc = acc / num_class
    aupr = aupr / num_class
    pre = pre / num_class
    rec = rec / num_class
    f1 = f1 / num_class
    
    print("|{:^12}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|{:^12.4f}|".format("Overall", acc, auc_, pre, rec, f1, kappa, aupr))

    return acc, auc_, rec, pre, f1, kappa, aupr

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetuning VisionFM while training a multi-class classifier on top')
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.""")
    parser.add_argument('--arch', default='vit_base_mb_vsl_txt', type=str, help='Architecture.')
    parser.add_argument('--input_size', type=int, default=448, help='Input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--window_size', default=7, type=int, help='Window size of the model.')
    parser.add_argument('--pretrained_weights', default='pretrain/VFM_Fundus_weights.pth', type=str, help="""Path to pretrained 
        weights""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of finetuning.')
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the beginning of
        training the classifier""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='data/OCTA-Enface', type=str,
        help='Please specify path to the eye image data.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--task', default='inhouse', type=str)
    parser.add_argument('--extra', default='', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="output_dir", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=5, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume finetuning')

    parser.add_argument('--info_dim', default=768, type=int)
    parser.add_argument('--num_samples', type=int, default=800)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--eval', action='store_true', #default=True,
                        help='Perform evaluation only')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--order_path', type=str, default="img2order.pkl")
    parser.add_argument('--inverse_order_path', type=str, default="img2inverseorder.pkl")
    parser.add_argument('--text_path', type=str, default="vessel_descrp/p2res_disease.json")
    parser.add_argument("--temperature", default=0.07, type=float)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="vampire",
        name=args.task,
        config=vars(args)
    )

    args.output_dir = os.path.join(args.output_dir, args.task, args.fold)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Start finetuning {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
    wandb.finish()
