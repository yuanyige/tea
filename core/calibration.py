import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torchmetrics.functional import calibration_error
from core.setada import *

def plot_calibration_hist(model, x, y, batch_size = 100, logger=None, device = None, ada=None, if_adapt=True, if_vis=False, c=None, s=None, myfont=None):
    try:
        model.reset()
        logger.info("resetting model")
    except:
        logger.warning("not resetting model")
    
    if device is None:
        device = x.device
    n_bins=10
    n_batches = math.ceil(x.shape[0] / batch_size)
    
    bin_boundaries = np.linspace(0, 1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accs=0
    preds = []
    cons = []
    eces=[]
    mces=[]

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            
            if ada == 'energy':
                output, energes = model(x_curr, y_curr, if_adapt=if_adapt, counter=counter, if_vis=if_vis)
            else:
                output = model(x_curr)
            
            accs += (output.max(1)[1] == y_curr).float().sum()
            
            pred = torch.softmax(output, dim=1) 
            ece = calibration_error(pred, y_curr, norm='l1', task='multiclass', num_classes=10, n_bins=10)
            mce = calibration_error(pred, y_curr, norm='max', task='multiclass', num_classes=10, n_bins=10)
            confidences, predictions = pred.max(dim=1)
            
            preds.append(predictions.cpu().numpy())
            cons.append(confidences.cpu().numpy())
            eces.append(ece.detach().cpu())
            mces.append(mce.detach().cpu())

    preds = np.concatenate(preds)
    cons = np.concatenate(cons)

    accs = (accs / x.shape[0]).detach().cpu()
    eces = np.array(eces).mean()
    mces = np.array(mces).mean()

    bin_acc = []
    bin_num = []
    
    y = y.cpu().numpy()
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        
        # Including the upper boundary by a small epsilon for the last bin.
        epsilon = 1e-8 if bin_upper==1 else 0 
        indices = np.where((cons>bin_lower) & (cons<=bin_upper + epsilon))
        
        indices = indices[0]
        num_data = len(indices.tolist())
        print(num_data)
        if num_data == 0:
            acc = 0
        else:
            print((preds[indices] == y[indices]))
            print(preds[indices])
            print(y[indices])
            acc = ((preds[indices] == y[indices]).astype(float).sum())/num_data
        
        bin_acc.append(acc)
        bin_num.append(num_data)
        
    def thousands_formatter(x, pos):
        return '%1.0fk' % (x * 1e-3)
    
    fig, ax1 = plt.subplots(figsize=(6.6, 6))

    x = [i/10 for i in range(10)] 

    x2 = [-0.1]+x+[1.0]
    if ada == 'energy':
        name = "TEA"
    elif  ada == 'source':
        name = "Source"
    else:
        name = ada.upper()


    ax1.bar(x, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], color='grey', alpha=0.4, width=0.1, edgecolor='white', linewidth=0.5, align='edge', label='Gap')
    ax1.bar(x, bin_acc, color='#8080F5', width=0.1, edgecolor='black', linewidth=0.5, align='edge', label=name)
    
    ax1.plot(x2, [-0.05, 0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95, 1.05], color='k', linestyle='dashed', linewidth=2.5)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    ax2 = ax1.twinx()
    x3=[i+0.03 for i in x]
    ax2.bar(x3, bin_num, width=0.04, color='#E55050', align='edge',label="#Sample") 
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter))
    ax2.set_ylim([0,12000])
    
    ax2.set_ylabel('Number of Samples', fontsize = 19, fontproperties=myfont)
    ax1.set_xlabel("Confidence", fontsize = 19, fontproperties=myfont)
    ax1.set_ylabel("Accuracy", fontsize = 19, fontproperties=myfont)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2

    plt.legend(handles, labels, loc="upper left", ncol=3, prop=myfont2)
    plt.text(0.02, 10100, r"ECE($\downarrow$)={:.2f}%".format(eces*100),fontsize = 24,fontproperties=myfont)
    plt.text(0.02, 9100, r"MCE($\downarrow$)={:.2f}%".format(mces*100),fontsize = 24, fontproperties=myfont)
    plt.title(name,fontsize = 24,fontproperties=myfont)
    plt.tight_layout()
    plt.savefig("./save/others/calibration/{}-{}-{}.pdf".format(name,c,s), format='pdf')
    plt.close()



def calibration_ood(model, cfg, logger, device):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        for c in range(len(cfg.CORRUPTION.TYPE)):
            for s in range(len(cfg.CORRUPTION.SEVERITY)):
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
                x_test, y_test = load_data(cfg.CORRUPTION.DATASET+'c', cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, False,
                                            [cfg.CORRUPTION.TYPE[c]])
                x_test, y_test = x_test.to(device), y_test.to(device)
                plot_calibration_hist(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, c=cfg.CORRUPTION.TYPE[c], s=cfg.CORRUPTION.SEVERITY[s])


def calibration_ori(model, cfg, logger, device):
        if 'cifar' in cfg.CORRUPTION.DATASET:
            x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
            x_test, y_test = x_test.to(device), y_test.to(device)
            plot_calibration_hist(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, logger=logger, ada=cfg.MODEL.ADAPTATION, if_adapt=True, c='original', s=0)



