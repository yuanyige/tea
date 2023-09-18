import os
import logging

import torch
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


from core.eval import evaluate
from core.conf import cfg, load_cfg_fom_args
from core.utils import set_seed, set_logger
from core.setup.adapt import *

logger = logging.getLogger(__name__)

def main(description):
    load_cfg_fom_args(description)
    set_seed(cfg)
    set_logger(cfg)

    # configure base model
    if (cfg.CORRUPTION.DATASET == 'cifar10') or (cfg.CORRUPTION.DATASET == 'cifar100'):
        base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    elif cfg.CORRUPTION.DATASET == 'mnist':
        base_model = torch.load(os.path.join(cfg.CKPT_DIR, 'mnist', str(cfg.MODEL.ARCH)+'.pt')).cuda()
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
    # elif cfg.MODEL.ADAPTATION == "eta":
    #     logger.info("test-time adaptation: ETA")
    #     model = setup_eta(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "eata":
        logger.info("test-time adaptation: EATA")
        model = setup_eata(base_model, cfg, logger)
    elif cfg.MODEL.ADAPTATION == "energy":
        logger.info("test-time adaptation: ENERGY")
        model = setup_energy(base_model, cfg, logger)
    else:
        raise NotImplementedError
    
    # evaluate on each severity and type of corruption in turn
    evaluate(model, cfg, logger)

if __name__ == '__main__':
    main('"TTA evaluation.')
