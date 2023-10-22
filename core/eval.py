import numpy as np
import pandas as pd
import math

import torch
import torch.nn.functional as F

import robustbench
from core.setup.data import load_data, load_dataloader
from autoattack import AutoAttack


def clean_accuracy(model, x, y, batch_size = 100, device = None, ada=None, if_adapt=True, if_vis=False):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            
            if ada == 'source':
                output = model(x_curr)
            else:
                output = model(x_curr, if_adapt=if_adapt, counter=counter, if_vis=if_vis)

            acc += (output.max(1)[1] == y_curr).float().sum()

    return acc.item() / x.shape[0]


def clean_accuracy_loader(model, test_loader, logger=None):
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



def evaluate_ood(model, cfg, logger):
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100') or (cfg.CORRUPTION.DATASET == 'tin200'):
        res = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        res_ori = np.zeros((len(cfg.CORRUPTION.SEVERITY),len(cfg.CORRUPTION.TYPE)))
        for c in range(len(cfg.CORRUPTION.TYPE)):
            for s in range(len(cfg.CORRUPTION.SEVERITY)):
                # reset adaptation for each combination of corruption x severity
                # note: for evaluation protocol, but not necessarily needed
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
                x_test, y_test = load_data(cfg.CORRUPTION.DATASET+'c', cfg.CORRUPTION.NUM_EX,
                                            cfg.CORRUPTION.SEVERITY[s], cfg.DATA_DIR, False,
                                            [cfg.CORRUPTION.TYPE[c]])
                x_test, y_test = x_test.cuda(), y_test.cuda()
                acc = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, ada=cfg.MODEL.ADAPTATION, if_adapt=True)
                logger.info(f"acc % [{cfg.CORRUPTION.TYPE[c]}{cfg.CORRUPTION.SEVERITY[s]}]: {acc:.2%}")      
                res[s, c] = acc

                x_test_ori, y_test_ori = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
                x_test_ori, y_test_ori = x_test_ori.cuda(), y_test_ori.cuda()
                acc_ori = clean_accuracy(model, x_test_ori, y_test_ori, cfg.OPTIM.BATCH_SIZE, ada=cfg.MODEL.ADAPTATION, if_adapt=False)
                logger.info(f"ori acc: {acc_ori:.2%}")      
                res_ori[s, c] = acc_ori

        frame = pd.DataFrame({i+1: res[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))

        frame = pd.DataFrame({i+1: res_ori[i, :] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}, index=cfg.CORRUPTION.TYPE)
        frame.loc['average'] = {i+1: np.mean(res_ori, axis=1)[i] for i in range(0, len(cfg.CORRUPTION.SEVERITY))}
        frame['avg'] = frame[list(range(1, len(cfg.CORRUPTION.SEVERITY)+1))].mean(axis=1)
        logger.info("\n"+str(frame))
    
    elif cfg.CORRUPTION.DATASET == 'mnist':
        _, _, _, test_loader = load_dataloader(root=cfg.DATA_DIR, dataset=cfg.CORRUPTION.DATASET, batch_size=cfg.OPTIM.BATCH_SIZE, if_shuffle=False, logger=logger)
        acc = clean_accuracy_loader(model, test_loader, logger=logger)
        logger.info("Test set Accuracy: {}".format(acc))
    
    else:
        raise NotImplementedError

def evaluate_adv(base_model, model, cfg, logger):
        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        adversary = AutoAttack(base_model, norm='L2', eps=0.5, version='custom', attacks_to_run=['apgd-ce'])
        adversary.apgd.n_restarts = 1
        x_adv = adversary.run_standard_evaluation(x_test, y_test)
        acc = clean_accuracy(model, x_adv, y_test, cfg.OPTIM.BATCH_SIZE, if_adapt=True)
        print("acc",acc)


def evaluate_ori(model, cfg, logger):
        x_test, y_test = load_data(cfg.CORRUPTION.DATASET, n_examples=cfg.CORRUPTION.NUM_EX, data_dir=cfg.DATA_DIR)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        acc = clean_accuracy(model, x_test, y_test, cfg.OPTIM.BATCH_SIZE, ada=cfg.MODEL.ADAPTATION, if_adapt=True)
        logger.info("Test set Accuracy: {}".format(acc))


