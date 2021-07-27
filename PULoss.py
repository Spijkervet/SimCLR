from torch import nn
import torch


class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss#lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)
    
    def forward(self, inp, target, test=False):  
        # assert(inp.shape == target.shape)
        
        # inp1 = inp[:, 1]
        # inp0 = inp[:, 0] 
        inp = inp*2 - 1
        target = target*2 - 1 # else : target -1 == self.unlabeled in next row


        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = torch.tensor(self.prior).cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))
        
        y_positive = self.loss_func(positive*inp) * positive
        y_positive_inv = self.loss_func(-positive*inp) * positive
        y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

        # y_positive = self.loss_func(positive*inp1) * positive
        # y_positive_inv = self.loss_func(positive*inp0) * positive
        # y_unlabeled = self.loss_func(unlabeled*inp0) * unlabeled

        positive_risk = self.prior * torch.sum(y_positive)/ n_positive
        negative_risk = - self.prior *torch.sum(y_positive_inv)/ n_positive + torch.sum(y_unlabeled)/n_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk 
        else:
            return positive_risk+negative_risk