import torch
import torch.nn as nn

class MedianTripletHead(nn.Module):
    def __init__(self,# predictor, size_average=True
                 gamma=2):
        super(MedianTripletHead, self).__init__()
        # self.predictor = builder.build_neck(predictor)
        # self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=3.1) # actually, biggest margin possible should be 1+2*gamma
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
        dist = -2. * torch.matmul(pred_norm, target_norm.t()) 
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
