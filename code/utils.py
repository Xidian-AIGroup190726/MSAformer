import numpy as np
import torch
from torch.nn import functional as F
import random
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient
def train_model(model, train_loader, optimizer, epoch):
    loop = tqdm(train_loader, leave=True)
    model.train()
    correct = 0.0
def train_model(model, train_loader, optimizer, epoch, amp_mode="fp16", device=None):
    """
    amp_mode: "fp16" | "bf16" | "off"
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    use_amp = (amp_mode != "off")
    amp_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "off" : torch.float32
    }[amp_mode]
    scaler = GradScaler(enabled=(amp_mode == "fp16"))

    correct = 0
    total = 0
    loop = tqdm(train_loader, leave=True)

    for step, batch in enumerate(loop):
        ms, pan, label = batch[0], batch[1], batch[2]

        ms    = ms.to(device, non_blocking=True)
        pan   = pan.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            output = model(ms, pan)
            loss = F.cross_entropy(output, label)

        # 反向 + 更新
        if amp_mode == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()


        if step % 100 == 0 or step == len(loop) - 1:
            accuracy = correct * 100.0 /  len(train_loader.dataset)
            logging.info(f"Train Epoch: {epoch} Step: {step}  Accuracy: {accuracy:.2f}%")
            print(f"Train Epoch: {epoch} \t  Step: {step} \t Train Accuracy: {accuracy:.2f}%")

        loop.set_postfix(loss=loss.item(), epoch=epoch, accuracy=correct * 100.0 /  len(train_loader.dataset), mode='train')

    loop.close()
    print("Train Accuracy: {:.2f}%".format(accuracy))
    logging.info("Final Train Accuracy for Epoch {}: {:.2f}%".format(epoch, accuracy))


def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    f = open('result_4layer.txt', 'a')
    l = 0
    print('testing...')
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Test', ncols=130)
        for ms, pan, target, _ in loop:
            l += 1
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            output = model(ms, pan)
            test_loss += F.cross_entropy(output, target.long()).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
            if l == 1:
                y_pred = pred.cpu().numpy()
                y_true = target.cpu().numpy()
            else:
                y_pred = np.concatenate((y_pred, pred.cpu().numpy()), axis=0)
                y_true = np.concatenate((y_true, target.cpu().numpy()), axis=0)
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))
        con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print("混淆矩阵", con_mat)
        f.write("Confusion Matrix:\n")
        f.write(str(con_mat) + "\n")
        # 计算性能参数
        all_acr = 0
        p = 0
        column = np.sum(con_mat, axis=0)  # 列求和
        line = np.sum(con_mat, axis=1)  # 行求和
        for i, clas in enumerate(con_mat):
            precise = clas[i]
            all_acr = precise + all_acr
            acr = precise / column[i]
            recall = precise / line[i]
            f1 = 2 * acr * recall / (acr + recall)
            temp = column[i] * line[i]
            p = p + temp
            # print('PRECISION:',acr,'||RECALL:',recall,'||F1:',f1)#查准率 #查全率 #F1
            print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
            f.write("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f \n" % (i, acr, recall, f1))
        OA = np.trace(con_mat) / np.sum(con_mat)
        print('OA:', OA)


        AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
        print('AA:', AA)

        Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
        Kappa = (OA - Pc) / (1 - Pc)
        print('Kappa:', Kappa)
        torch.save(model, 'model_hohhot{}.pkl'.format(OA))
        f.write('OA:'+str(OA))
        f.write('AA:'+str(AA))
        f.write('KAPPA:'+str(Kappa)+'\n')
