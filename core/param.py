import torch.nn as nn
from copy import deepcopy

def collect_params(model, ada_param=['bn'], logger=None):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []

    if 'all' in  ada_param:
        logger.info('adapting all weights')
        return model.parameters(), 'all'
    
    if 'bn' in ada_param:
        logger.info('adapting weights of batch-normalization layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d): 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    if 'gn' in ada_param:
        logger.info('adapting weights of group-normalization layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.GroupNorm): 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    
    if 'in' in ada_param:
        logger.info('adapting weights of instance-normalization layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.InstanceNorm2d): 
                for np, p in m.named_parameters():
                    print(np)
                    exit(0)
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    if 'conv' in ada_param:
        logger.info('adapting weights of conv layer')
        for nm, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
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


def configure_model(model, ada_param=None):
    """Configure model for use with tent."""

    if 'all' in  ada_param:
        return model

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

    if 'gn' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.GroupNorm):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    
    if 'in' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.InstanceNorm2d): 
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    if 'conv' in ada_param:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad_(True)

    if 'fc' in ada_param:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)