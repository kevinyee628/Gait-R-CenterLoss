import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    # feat_dim = 3
    def __init__(self, margin, num_classes=10, feat_dim=3, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # self.margin = margin
        self.use_gpu = use_gpu
        self.margin = margin

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        n, m, d = x.size()
        hp_mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).byte().view(-1)
        hn_mask = (labels.unsqueeze(1) != labels.unsqueeze(2)).byte().view(-1)

        batch_size = x.size(0)
        dist = self.basenet(x)
        mean = dist.mean(1).mean(1)
        dist = dist.view(-1)

        # TODO:降低feature维度
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)

        distmat = torch.pow(full_hp_dist, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(1, -2, full_hp_dist, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        # TODO:降为1D
        # full_loss_metric = F.relu(self.margin + dist).view(n, -1)

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def batch_dists(self, x, ):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist
