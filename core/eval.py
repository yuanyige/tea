import numpy as np
import pandas as pd
import math

import torch
import torch.nn.functional as F

import robustbench
from robustbench.data import load_cifar10c, load_cifar10
from core.setup.data import load_data


def evaluate(model, cfg, logger):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100'):
        res = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        for c in range(len(cfg.CORRUPTION.TYPE)):
            for s in range(len(cfg.CORRUPTION.SEVERITY)):
                # reset adaptation for each combination of corruption x severity
                # note: for evaluation protocol, but not necessarily needed
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
                x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, False,
                                            [cfg.CORRUPTION.TYPE[c]])
                # x_test, y_test = load_cifar10(cfg.CORRUPTION.NUM_EX, cfg.DATA_DIR)
                x_test, y_test = x_test.cuda(), y_test.cuda()
                acc = robustbench.utils.clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE)
                #err = 1. - acc
                logger.info(f"error % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {acc:.2%}")      
                res[s, c] = acc
        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))
    
    elif cfg.CORRUPTION.DATASET == 'mnist':
        _, _, _, test_loader = load_data(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=None)
        acc = clean_accuracy(model, test_loader, logger=logger)
        logger.info("Test set Accuracy: {}".format(acc))
    
    else:
        raise NotImplementedError


def clean_accuracy(model, test_loader, logger=None):
    test_loss = 0
    correct = 0
    index = 1
    total_step = math.ceil(len(test_loader.dataset) / test_loader.batch_size)
    with torch.no_grad():
        for data, target in test_loader:
            logger.info("Test Batch Process: {}/{}".format(index, total_step))
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
            test_loss += F.cross_entropy(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            index = index + 1
            
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc


