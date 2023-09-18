"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight. 
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.jit

import math
import torch.nn.functional as F
from core.setup.optim import load_model_and_optimizer, copy_model_and_optimizer

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class EATA(nn.Module):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, fisher_alpha=2000.0, steps=1, episodic=False, e_margin=math.log(1000)/2-1, d_margin=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EATA requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = e_margin # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = d_margin # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)

        self.current_model_probs = None # the moving average of probability vector (Eqn. 4)

        self.fishers = fishers # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = fisher_alpha # trade-off \beta for two losses (Eqn. 8) 

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, num_counts_2, num_counts_1, updated_probs = forward_and_adapt_eata(x, self.model, self.optimizer, self.fishers, self.e_margin, self.current_model_probs, fisher_alpha=self.fisher_alpha, num_samples_update=self.num_samples_update_2, d_margin=self.d_margin)
                self.num_samples_update_2 += num_counts_2
                self.num_samples_update_1 += num_counts_1
                self.reset_model_probs(updated_probs)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_eata(x, model, optimizer, fishers, e_margin, current_model_probs, fisher_alpha=50.0, d_margin=0.05, scale_factor=2, num_samples_update=0):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    Return: 
    1. model outputs; 
    2. the number of reliable and non-redundant samples; 
    3. the number of reliable samples;
    4. the moving average  probability vector over all previous samples
    """
    # forward
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    ids1 = filter_ids_1
    ids2 = torch.where(ids1[0]>-0.1)
    entropys = entropys[filter_ids_1] 
    # filter redundant samples
    if current_model_probs is not None: 
        cosine_similarities = F.cosine_similarity(current_model_probs.unsqueeze(dim=0), outputs[filter_ids_1].softmax(1), dim=1)
        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < d_margin)
        entropys = entropys[filter_ids_2]
        ids2 = filter_ids_2
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
    else:
        updated_probs = update_model_probs(current_model_probs, outputs[filter_ids_1].softmax(1))
    coeff = 1 / (torch.exp(entropys.clone().detach() - e_margin))
    # implementation version 1, compute loss, all samples backward (some unselected are masked)
    entropys = entropys.mul(coeff) # reweight entropy losses for diff. samples
    loss = entropys.mean(0)
    """
    # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
    # if x[ids1][ids2].size(0) != 0:
    #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
    """
    if fishers is not None:
        ewc_loss = 0
        for name, param in model.named_parameters():
            if name in fishers:
                ewc_loss += fisher_alpha * (fishers[name][0] * (param - fishers[name][1])**2).sum()
        loss += ewc_loss
    if x[ids1][ids2].size(0) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()
    return outputs, entropys.size(0), filter_ids_1[0].size(0), updated_probs


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)



def transform(im_sz):
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tr_transforms = transforms.Compose([transforms.RandomResizedCrop(im_sz),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                        transforms.ToTensor()])
    te_transforms = transforms.Compose([transforms.Resize(im_sz),
                                        transforms.CenterCrop(im_sz),
                                        transforms.ToTensor()])
    te_transforms_imageC = transforms.Compose([transforms.CenterCrop(im_sz),transforms.ToTensor()])

    rotation_tr_transforms = tr_transforms
    rotation_te_transforms = te_transforms
    return te_transforms, te_transforms_imageC, rotation_te_transforms

class ImagePathFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform):
		super(ImagePathFolder, self).__init__(traindir, train_transform)	

	def __getitem__(self, index):
		path, _ = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		path, pa = os.path.split(path)
		path, pb = os.path.split(path)
		return img, 'val/%s/%s' %(pb, pa)

# =========================Rotate ImageFolder Preparations Start======================
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


# =========================Rotate ImageFolder Preparations End======================

# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created. 
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def prepare_test_data(root, corruption, use_transforms, batch_size, im_sz):
    te_transforms, te_transforms_imageC, rotation_te_transforms = transform(im_sz)
    if corruption == 'original':
        te_transforms_local = te_transforms if use_transforms else None
    else:
        #te_transforms_local = te_transforms_imageC if use_transforms else None
        raise
    
    if corruption == 'original':
        print('Test on the original test set')
        validdir = os.path.join(root, 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False, rotation_transform=rotation_te_transforms)
    else:
        # print('Test on %s level %d' %(corruption, level))
        # validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        # teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False, rotation_transform=rotation_te_transforms)
        raise
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False num_workers=4, pin_memory=True)
    return teset, teloader

# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.
#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names


# def copy_model_and_optimizer(model, optimizer):
#     """Copy the model and optimizer states for resetting after adaptation."""
#     model_state = deepcopy(model.state_dict())
#     optimizer_state = deepcopy(optimizer.state_dict())
#     return model_state, optimizer_state


# def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
#     """Restore the model and optimizer states from copies."""
#     model.load_state_dict(model_state, strict=True)
#     optimizer.load_state_dict(optimizer_state)


# def configure_model(model):
#     """Configure model for use with eata."""
#     # train mode, because eata optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what eata updates
#     model.requires_grad_(False)
#     # configure norm for eata updates: enable grad + force batch statisics
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#     return model


# def check_model(model):
#     """Check model for compatability with eata."""
#     is_training = model.training
#     assert is_training, "eata needs train mode: call model.train()"
#     param_grads = [p.requires_grad for p in model.parameters()]
#     has_any_params = any(param_grads)
#     has_all_params = all(param_grads)
#     assert has_any_params, "eata needs params to update: " \
#                            "check which require grad"
#     assert not has_all_params, "eata should not update all params: " \
#                                "check which require grad"
#     has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
#     assert has_bn, "eata needs normalization for its optimization"