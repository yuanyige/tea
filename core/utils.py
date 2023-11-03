import os
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .resnet import  ResNet, Bottleneck, BasicBlock

def set_seed(cfg):
    os.environ['PYTHONHASHSEED'] =str(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

def set_logger(cfg):
    os.makedirs(cfg.SAVE_DIR,exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def train_base(epoch, model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def build_model_res50gn(group_norm, num_classes):
    print('Building model...')
    def gn_helper(planes):
        return nn.GroupNorm(group_norm, planes)
    net = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=num_classes, norm_layer=gn_helper)
    return net

def build_model_res18bn(num_classes):
    print('Building model...')
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=nn.BatchNorm2d)


# def build_model(group_norm, depth, num_classes):
#     print('Building model...')
#     def gn_helper(planes):
#         return nn.GroupNorm(group_norm, planes)
#     net = ResNet(depth, 1, channels=3, classes=num_classes, norm_layer=gn_helper)
#     # if hasattr(args, 'parallel') and args.parallel:
#     #     net = torch.nn.DataParallel(net)
#     return net

# def run():
#     train_loader, test_loader = load_dataset('mnist')
#     # 加载模型并修改第一层以适应MNIST输入尺寸
#     model = resnet18(pretrained=False, num_classes=10)
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     model.maxpool = torch.nn.Identity()  # 移除maxpool层，因为MNIST图像尺寸较小
#     model.cuda()

#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
#     # 实际进行训练和测试
#     for epoch in range(1, 20 + 1):  # 总共进行20轮训练
#         train(epoch, model, train_loader, optimizer, criterion)
#         test(model, test_loader, criterion)
    
#     # 保存整个模型
#     torch.save(model, "mnist_resnet18.pt")



#     # def calculate_fid():
# #     fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
# #                                                     inception_model,
# #                                                     transform=transform)
#     # return fid_value