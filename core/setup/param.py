import torch.nn as nn

def collect_params(model, ada_param=['bn'], logger=None):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    if 'bn' in ada_param:
        logger.info('adapting weights of batch-normalization layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    if 'fc' in ada_param:
        logger.info('adapting weights of fully-connected layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.Linear):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    return params, names

def configure_model(model, ada_param=['bn']):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    if 'bn' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    if 'fc' in ada_param:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.requires_grad_(True)
    return model