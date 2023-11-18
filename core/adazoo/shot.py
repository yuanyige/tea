import copy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from core.param import load_model_and_optimizer, copy_model_and_optimizer


class SHOT(nn.Module):
    """
    "Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation"
    """
    def __init__(self, algorithm, optimizer, steps, threshold, clf_coeff):
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        theta (float) : clf coefficient
        gamma (int) : number of updates
        """
        super().__init__()
        # self.model, self.optimizer = self.configure_model_optimizer(algorithm, alpha=alpha, lr=lr, wd=wd)
        self.model, self.optimizer = algorithm,  optimizer
        self.beta = threshold
        self.theta = clf_coeff
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
        
        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()
        return outputs
    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        # (3) pseudo label
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])

        loss = ent_loss + self.theta * clf_loss
        return loss

    # def configure_model_optimizer(self, algorithm, alpha, lr, wd):
    #     adapted_algorithm = copy.deepcopy(algorithm)
    #     optimizer = torch.optim.Adam(
    #         adapted_algorithm.parameters(), 
    #         # adapted_algorithm.classifier.parameters(), 
    #         lr = lr  * alpha,
    #         weight_decay = wd
    #     )
    #     return adapted_algorithm, optimizer

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

class SHOTIM(SHOT):    
    def loss(self, outputs):
        # (1) entropy
        ent_loss = softmax_entropy(outputs).mean(0)

        # (2) diversity
        softmax_out = F.softmax(outputs, dim=-1)
        msoftmax = softmax_out.mean(dim=0)
        ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

        return ent_loss
           
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)