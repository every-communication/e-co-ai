import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = sum([torch.sum(pred[:, i] == target).item() for i in range(k)])
    return correct / len(target)

def precision(output, target, average='macro'):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        true_positive = (pred == target) & (target == 1)
        predicted_positive = pred == 1
        precision = true_positive.sum().item() / (predicted_positive.sum().item() + 1e-6)
    return precision

def recall(output, target, average='macro'):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        true_positive = (pred == target) & (target == 1)
        actual_positive = target == 1
        recall = true_positive.sum().item() / (actual_positive.sum().item() + 1e-6)
    return recall

def f1_score(output, target, average='macro'):
    with torch.no_grad():
        prec = precision(output, target, average)
        rec = recall(output, target, average)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-6)
    return f1

def confusion_matrix_metric(output, target, num_classes):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        cm = confusion_matrix(target.cpu(), pred.cpu(), labels=range(num_classes))
    return cm