from copy import deepcopy
import os
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from core.param import load_model_and_optimizer, copy_model_and_optimizer

def init_random(bs, im_sz=32, n_ch=3):
    return torch.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits
        

def sample_p_0(reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    if len(replay_buffer) == 0:
        return init_random(bs, im_sz=im_sz, n_ch=n_ch), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds

    buffer_samples = replay_buffer[inds]
    random_samples = init_random(bs, im_sz=im_sz, n_ch=n_ch)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz, n_ch=n_ch, device=device ,y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples, init_samples.detach()

class Energy(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 buffer_size=10000, sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, if_cond=False, 
                 n_classes=10, im_sz=32, n_ch=3, path=None, logger=None): 
        super().__init__()

        self.energy_model=EnergyModel(model)
        self.replay_buffer = init_random(buffer_size, im_sz=im_sz, n_ch=n_ch)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond
        
        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        
        self.path=path
        self.logger = logger

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()
        
        if if_adapt:
            for i in range(self.steps):
                outputs = forward_and_adapt(x, self.energy_model, self.optimizer, 
                                            self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std, self.reinit_freq,
                                            if_cond=self.if_cond, n_classes=self.n_classes)
                if i % 1 == 0 and if_vis:
                    visualize_images(path=self.path, replay_buffer_old=self.replay_buffer_old, replay_buffer=self.replay_buffer, energy_model=self.energy_model, 
                                    sgld_steps=self.sgld_steps, sgld_lr=self.sgld_lr, sgld_std=self.sgld_std, reinit_freq=self.reinit_freq,
                                    batch_size=100, n_classes=self.n_classes, im_sz=self.im_sz, n_ch=self.n_ch, device=x.device, counter=counter, step=i)
        else:
            self.energy_model.eval()
            with torch.no_grad():
                outputs = self.energy_model.classify(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
@torch.enable_grad()
def visualize_images(path, replay_buffer_old, replay_buffer, energy_model, 
                     sgld_steps, sgld_lr, sgld_std, reinit_freq,
                     batch_size, n_classes, im_sz, n_ch, device=None, counter=None, step=None):
    num_cols=10
    repeat_times = batch_size // n_classes
    y = torch.arange(n_classes).repeat(repeat_times).to(device) 
    x_fake, _ = sample_q(energy_model, replay_buffer, n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)
    images = x_fake.detach().cpu()
    save_image(images , os.path.join(path, 'sample.png'), padding=2, nrow=num_cols)

    num_cols=40
    images_init = replay_buffer_old.cpu()
    images = replay_buffer.cpu() 
    images_diff = replay_buffer.cpu() - replay_buffer_old.cpu()
    if step == 0:
        save_image(images_init , os.path.join(path, 'buffer_init.png'), padding=2, nrow=num_cols)
    save_image(images , os.path.join(path, 'buffer-'+str(counter)+"-"+str(step)+'.png'), padding=2, nrow=num_cols) # 
    save_image(images_diff , os.path.join(path, 'buffer_diff.png'), padding=2, nrow=num_cols)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, if_cond=False, n_classes=10):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    batch_size=x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device
    
    if if_cond == 'uncond':
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    # adapt
    loss = (- (energy_real - energy_fake)) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)

    return outputs
