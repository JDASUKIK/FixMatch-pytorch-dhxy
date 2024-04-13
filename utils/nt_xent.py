import torch
import torch.nn as nn
import torch.distributed as dist
# from .gather import GatherLayer


class NT_Xent(nn.Module):

    def __init__(self, batch_size, temperature, world_size):
        """
            Args:
                temperature: the τ value of NT_Xent Loss
                world_size: device number used for distributed training.
            """
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        # 对角线设为False
        mask = mask.fill_diagonal_(0)
        # 将第i个样本与i+batch_size即其对应的正样本设为False（即两条副对角线）
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017),
            we treat the other 2(N − 1) augmented examples within a minibatch
            as negative examples.
        """
        self.batch_size = z_i.shape[0]
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # if self.world_size > 1:
            # z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)    # 获得上面一条副对角线
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)   # 获得下面一条副对角线

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        # 将正样本对得分放到第一排，使得labels全为0
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
