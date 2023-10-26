import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


from core.eval import evaluate_ori, evaluate_ood
from core.conf import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger, build_model_res50gn, build_model_res18bn
from core.setup.adapt import *

logger = logging.getLogger(__name__)

def main(description):
    load_cfg_fom_args(description)
    set_seed(cfg)
    set_logger(cfg)

    device = torch.device('cpu')

    # configure base model
    if 'GN' in cfg.MODEL.ARCH:
        base_model = build_model_res50gn(8, cfg.CORRUPTION.NUM_CLASSES).to(device)
        ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/ResNet50G.pth'.format(cfg.CORRUPTION.DATASET)))
        base_model.load_state_dict(ckpt['state_dict'])
    else:
        if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100' and cfg.MODEL.ARCH != 'Standard'):
            base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).to(device)
        elif (cfg.CORRUPTION.DATASET == 'mnist')or (cfg.CORRUPTION.DATASET == 'tin200') or (cfg.CORRUPTION.DATASET == 'cifar100' and cfg.MODEL.ARCH == 'Standard'):
            base_model = torch.load(os.path.join(cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, str(cfg.MODEL.ARCH)+'.pt')).to(device)
        elif cfg.CORRUPTION.DATASET == 'pacs':
            base_model = build_model_res18bn(cfg.CORRUPTION.NUM_CLASSES).to(device)
            ckpt = torch.load(os.path.join(cfg.CKPT_DIR ,'{}/{}.pth'.format(cfg.CORRUPTION.DATASET,cfg.MODEL.ARCH)))
            base_model.load_state_dict(ckpt['state_dict'])
        else:
            raise NotImplementedError

    # configure tta model
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eta":
        logger.info("test-time adaptation: ETA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eata":
        logger.info("test-time adaptation: EATA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "sar":
        logger.info("test-time adaptation: SAR")
        model = setup_sar(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    # evaluate on each severity and type of corruption in turn
    # evaluate_adv(base_model, model, cfg, logger, device)
    evaluate_ood(model, cfg, logger, device)
    evaluate_ori(model, cfg, logger, device)
    
    

if __name__ == '__main__':
    main('"TTA evaluation.')
