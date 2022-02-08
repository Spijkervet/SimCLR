import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


# import torch
# import torch.nn as nn
# import torch.distributed as dist
# from .gather import GatherLayer
class PU_classif_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Linear1 = nn.Linear(input_size, hidden_size)
        self.ReLu = nn.LeakyReLU(0.01)
        self.Linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.Linear2(self.ReLu(self.Linear1(x)))


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
        if not 0 <= prior < 1:
            raise NotImplementedError("The class prior should be in [0, 1)")
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
        self.weight_onnpu = torch.tensor(1.0).cuda()
        # trainable linear Layer for mapping latent variables to 1d classification output for nnPU loss
        self.ClassifModel = PU_classif_model(latent_size, 512).cuda()

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
        N = len(target)
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
            return (-self.gamma * negative_risk) / N
        else:
            return (positive_risk + negative_risk) / N


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
        # onnpu_l = 0.5*(self.onnpu_loss(self.linear_classif(h_i), target) + self.onnpu_loss(self.linear_classif(h_j), target))
        pred_hi = self.ClassifModel(h_i)
        pred_hj = self.ClassifModel(h_j)
        onnpu_l = self.onnpu_loss(torch.vstack((pred_hi, pred_hj)), torch.hstack((target, target)))
        nt_xent_l = self.nt_xent_loss(z_i, z_j)

        loss = self.weight_onnpu*onnpu_l + nt_xent_l #0.8, 0.95
        return loss, self.ClassifModel, self.weight_onnpu


class MedianTripletHead(nn.Module):
    def __init__(self,# predictor, size_average=True
                 gamma=2):
        super(MedianTripletHead, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=2.) # actually, biggest margin possible should be 1+2*gamma
        self.gamma = gamma

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -1. * torch.matmul(pred_norm, target_norm.t()) # -2. *
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].median().unsqueeze(0))
            #down_k = torch.topk(dist[i][mask[i]==0], 5, dim=-1, largest=False)
            #down_k = down_k[0][-1].unsqueeze(0)
            #dist_an.append(down_k)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return loss_triplet # dict(loss=loss_triplet)

class SmoothTripletHead(nn.Module):
    def __init__(self,# predictor, size_average=True,
                 k, gamma=1):
        super(SmoothTripletHead, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin= 2.) # actually, biggest margin possible should be 1+gamma # changed
        self.gamma = gamma
        self.k = k

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -1. * torch.matmul(pred_norm, target_norm.t()) # -2.*
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0).repeat(self.k))
            down_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=False)
            down_k = down_k[0] # [-1].unsqueeze(0)
            dist_an.append(down_k)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return loss_triplet # dict(loss=loss_triplet)


class TripletNNPULoss(nn.Module):
    def __init__(self,# predictor, size_average=True,
                 prior, k, C=0, gamma=1):
        super(TripletNNPULoss, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        # self.ranking_loss = nn.MarginRankingLoss(margin=2)
        self.gamma = gamma
        self.k = k
        self.C = C
        self.prior = prior

    # def init_weights(self, init_linear='normal'):
    #     self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = input # self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = (-1. * torch.matmul(pred_norm, target_norm.t()) +1)/2 #-2.*
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())

        #dist_ap = []
        losses = []
        for i in range(n):

            dist_ap = (dist[i][mask[i]].max()) #.unsqueeze(0).repeat(k))
           # dist_ap = torch.cat(dist_ap)

            down_k, idx_down_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=False)
            # down_k = down_k[0] # [-1].unsqueeze(0)
            up_k, idx_up_k = torch.topk(dist[i][mask[i]==0], self.k, dim=-1, largest=True)
            # up_k = up_k[0] # [-1].unsqueeze(0)

            dist_pos_unl = torch.cat((down_k, up_k))

            # maybe redefine selection of unlabeled samples and esp. calculation of center
            feat_unl = torch.cat((target_norm[mask[i]==0][idx_down_k], target_norm[mask[i]==0][idx_up_k])) #, 
            center_unl = torch.mean(target_norm, dim=0)
            dist_unl = (-1. * torch.matmul(feat_unl, center_unl)  + 1)/2

        # y = torch.ones_like(dist_an)
        # loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)

            prior = self.prior # can be changed
            prior_prime = 0.5 # can be changed
            n_positive = 1
            n_unlabeled = 2*self.k

            positive_risk = prior_prime/n_positive * dist_ap
            # negative_risk =  (1-prior_prime)/(n_unlabeled*(1-prior)) * torch.sum(dist_unl) - ((1-prior_prime)*prior/(n_positive*(1-prior))) *torch.sum(dist_pos_unl) 
            negative_risk =  - ((1-prior_prime)/(n_positive * n_unlabeled )) *torch.sum(dist_pos_unl) 

            if negative_risk < self.C: #< -self.beta and self.nnPU:
                loss_n = -self.gamma * negative_risk 
            else:
                loss_n = positive_risk + negative_risk

            losses.append(loss_n.unsqueeze(dim=0))
        loss = torch.cat(losses).mean()

        return loss # dict(loss=loss_triplet)

class HeadNNPU(nn.Module):
    def __init__(self, prior, prior_prime=0.5,
                 loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=True, latent_size=2048):
        super(HeadNNPU, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        # self.ranking_loss = nn.MarginRankingLoss(margin=2)
        self.gamma = gamma
        self.prior = prior
        self.prior_prime = 0.5
        self.latent_size = latent_size
        self.predictor = nn.Linear(latent_size, 1)
        # torch.nn.init.constant_(self.predictor.weight, -1.)
        self.prior = prior
        self.prior_prime = prior_prime
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss #lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = -1
        self.min_count = torch.tensor(1.)

    def onnpu_loss(self, inp, target, prior=None, prior_prime=None):
    # assert(inp.shape == target.shape)
       # N = len(target)
        if prior is None:
            prior=self.prior
        if prior_prime is None:
            prior_prime=self.prior_prime
       # target = target*2 - 1 # else : target -1 == self.unlabeled in next row #!!!! -1 instead of 0!!

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        # if inp.is_cuda:
        #     self.min_count = self.min_count.cuda()
        #     prior = torch.tensor(prior).cuda()
        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(unlabeled))

        
        y_positive = self.loss_func(positive*inp) * positive
        y_positive_inv = self.loss_func(-positive*inp) * positive
        y_unlabeled = self.loss_func(-unlabeled*inp) * unlabeled

       # print(y_positive, y_positive_inv, y_unlabeled)

        positive_risk = prior_prime/n_positive * torch.sum(y_positive)
        negative_risk =  (1-prior_prime)/(n_unlabeled*(1-prior)) * torch.sum(y_unlabeled) - ((1-prior_prime)*prior/(n_positive*(1-prior))) *torch.sum(y_positive_inv)

        print(positive_risk, negative_risk)

        if negative_risk < -self.beta and self.nnPU:
            return (-self.gamma * negative_risk) #/ N
            
        else:
            return (positive_risk + negative_risk) #/ N
            

    def forward(self, z_i, z_j):
        """Forward head.
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size = z_i.size(0)
        pred = torch.flatten(self.predictor(torch.cat((z_i, z_j))))
        
        # pred_norm = nn.functional.normalize(pred, dim=1)
        # target_norm = nn.functional.normalize(target, dim=1)
        n = pred.size(0)
        # dist = (-1. * torch.matmul(pred_norm, target_norm.t()) +1)/2 #-2.*
        # idx = torch.arange(n)
        # mask = idx.expand(n, n).eq(idx.expand(n, n).t())

        targets = -1 * torch.ones((n, n), dtype=bool)
        targets = targets.fill_diagonal_(1)
        for i in range(batch_size): # * world_size
            targets[i, batch_size + i] = 1 # * world_size 
            targets[batch_size + i, i] = 1 # * world_size

        losses = []
        for i in range(n):
           # print(pred, targets[i])
            loss_n = self.onnpu_loss(pred, targets[i])
            #print(loss_n)
            losses.append(loss_n.unsqueeze(dim=0))
        loss = torch.cat(losses).mean()

        return loss 