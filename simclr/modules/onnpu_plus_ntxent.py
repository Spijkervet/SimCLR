import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from .gather import GatherLayer
class PU_plus_NTXent(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, batch_size, temperature, world_size, prior, prior_prime=0.5,
                 loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=True, latent_size=2048):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.prior_prime = prior_prime
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss #lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)

        # trainable weight parameter for weighting sum over OversamplednnPU Loss and NTXent Loss
        self.weight_onnpu = nn.Parameter(torch.tensor(0.5))
        # trainable linear Layer for mapping latent variables to 1d classification output for nnPU loss
        self.linear_classif = nn.Linear(latent_size, 1)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask
    
    def onnpu_loss(self, inp, target, prior=None, prior_prime=None):
        # assert(inp.shape == target.shape)
        if prior is None:
            prior=self.prior
        if prior_prime is None:
            prior_prime=self.prior_prime
        target = target*2 - 1 # else : target -1 == self.unlabeled in next row #!!!! -1 instead of 0!!

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        # if inp.is_cuda:
        #     self.min_count = self.min_count.cuda()
        #     prior = torch.tensor(prior).cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))
        
        y_positive = self.loss_func(positive*inp) * positive
        y_positive_inv = self.loss_func(-positive*inp) * positive
        y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

        positive_risk = prior_prime/n_positive * torch.sum(y_positive)
        negative_risk =  (1-prior_prime)/(n_unlabeled*(1-prior)) * torch.sum(y_unlabeled) - ((1-prior_prime)*prior/(n_positive*(1-prior))) *torch.sum(y_positive_inv)

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk 
        else:
            return positive_risk + negative_risk


    def nt_xent_loss(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward(self, h_i, h_j, z_i, z_j, target, prior=None, prior_prime=None):
        onnpu_l = 0.5*(self.onnpu_loss(self.linear_classif(h_i), target) + self.onnpu_loss(self.linear_classif(h_j), target))
        nt_xent_l = self.nt_xent_loss(z_i, z_j)

        loss = self.weight_onnpu*onnpu_l + (1-self.weight_onnpu)*nt_xent_l
        return loss
