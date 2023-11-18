import copy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from core.param import load_model_and_optimizer, copy_model_and_optimizer


class PseudoLabel(nn.Module):
    def __init__(self, algorithm, steps, threshold, alpha, lr, wd):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        super().__init__()
        self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=alpha, lr=lr, wd=wd)
        self.beta = threshold
        self.steps = steps
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = False
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if if_adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                # if self.hparams['cached_loader']:
                #     outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                # else:
                self.model.eval()
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                self.model.train()
        else:
            # if self.hparams['cached_loader']:
            #     outputs = self.model.classifier(x)
            # else:
            outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        
        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        optimizer.step()
        return outputs

    def configure_model_optimizer(self, algorithm, alpha, lr, wd):
        adapted_algorithm = copy.deepcopy(algorithm)
        optimizer = torch.optim.Adam(
            adapted_algorithm.parameters(), 
            lr = lr  * alpha,
            weight_decay = wd
        )
        return adapted_algorithm, optimizer

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)