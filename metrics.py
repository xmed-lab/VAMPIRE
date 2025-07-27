from scipy.spatial.distance import directed_hausdorff
from collections import *
import pandas as pd
from statistics import mean
import numpy as np

class MetricsStatistics:
    def __init__(self):
        self.epsilon = 1e-6
        self.func_dct = {
            "Precision": self.cal_precision,
            "Recall": self.cal_recall,
            "Specificity": self.cal_specificity,
            "Jaccard": self.cal_jaccard_index,
            "Dice": self.cal_dice,
            "IoU": self.cal_iou,
            "Hausdorff": self.cal_hausdorff
        }

    def cal_confusion_matrix(self, pred, label):
        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 0) & (label == 1)).sum().item()
        FN = ((pred == 1) & (label == 0)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()
        return TP, FP, FN, TN

    def cal_precision(self, preds, labels):
        precisions = []
        for pred, label in zip(preds, labels):
            TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
            pre = TP / (TP + FP + self.epsilon)
            precisions.append(pre)
        return np.mean(precisions)

    def cal_recall(self, preds, labels):
        recalls = []
        for pred, label in zip(preds, labels):
            TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
            rec = TP / (TP + FN + self.epsilon)
            recalls.append(rec)
        return np.mean(recalls)

    def cal_specificity(self, preds, labels):
        specs = []
        for pred, label in zip(preds, labels):
            TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
            spec = TN / (TN + FP + self.epsilon)
            specs.append(spec)
        return np.mean(specs)

    def cal_jaccard_index(self, preds, labels):
        jaccards = []
        for pred, label in zip(preds, labels):
            intersection = (pred & label).sum().item()
            union = (pred | label).sum().item()
            jaccard_index = intersection / (union + self.epsilon)
            jaccards.append(jaccard_index)
        return np.mean(jaccards)
    
    def cal_dice(self, preds, labels):
        dices = []
        for pred, label in zip(preds, labels):
            intersection = (pred & label).sum().item()
            union = pred.sum().item() + label.sum().item()
            dice = 2 * intersection / (union + self.epsilon)
            dices.append(dice)
        return np.mean(dices)
    
    def cal_iou(self, preds, labels):
        ious = []
        for pred, label in zip(preds, labels):
            TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
            ious.append(TP / (FN + FP + TP + self.epsilon))
        return np.mean(ious)

    def cal_hausdorff(self, preds, labels):
        hs = []
        for pred, label in zip(preds, labels):
            array1 = pred
            array2 = label
            dist1 = directed_hausdorff(array1, array2)[0]
            dist2 = directed_hausdorff(array2, array1)[0]
            hausdorff_dist = max(dist1, dist2)
            hs.append(hausdorff_dist)
        return np.mean(hs)
    