# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GeodesicLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GeodesicLoss, self).__init__()
        self.reduction = reduction

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1E-6, max=1-1E-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2):
        loss = self.compute_geodesic_distance(m1, m2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {self.reduction}')


class BCE_Loss(nn.Module):
    def __init__(self, args=None):
        super(BCE_Loss, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target, reduce="mean")
        return final_loss

class weight_Loss(nn.Module):
    def __init__(self, args=None):
        super(weight_Loss, self).__init__()
    def forward(self, weight_f):
        weight_loss_div = torch.mean(weight_f[:, :, 0]*weight_f[:, :, 1])
        weight_loss_gap = torch.mean(-torch.log(torch.max(weight_f[:, :, 0], dim=1)[0] - torch.min(weight_f[:, :, 0], dim=1)[0]))
        return weight_loss_div, weight_loss_gap    
    

class HuberLoss(nn.Module):
    def __init__(self, beta=0.1, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss(outputs / self.beta, targets / self.beta, reduction=self.reduction) * self.beta
        return final_loss
    

class KLLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLLoss, self).__init__()
        self.reduction = reduction

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean() if self.reduction == "mean" else div

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti(nn.Module):
    def __init__(self):
        super(KLLossMulti, self).__init__()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"


class REGLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(REGLoss, self).__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss((outputs / self.beta, targets / self.beta) * self.beta)
        return final_loss    


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, outputs, targets):
        final_loss = F.l2_loss(outputs, targets)
        return final_loss    





class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, text_projection, vq0_projection):
        """
        Computes the InfoNCE contrastive loss along the time axis.
        
        Args:
            text_projection: Tensor of shape (bs, 300, 512)
            vq0_projection: Tensor of shape (bs, 300, 512)
        
        Returns:
            Scalar loss value
        """
        bs, time_steps, dim = text_projection.shape
        
        # Normalize embeddings
        text_projection = F.normalize(text_projection, dim=-1)
        vq0_projection = F.normalize(vq0_projection, dim=-1)
        
        # Compute similarity matrix (bs, 300, 300)
        similarity_matrix = torch.bmm(text_projection, vq0_projection.transpose(1, 2)) / self.temperature
        
        # Create labels (identity matrix for positive pairs)
        labels = torch.arange(time_steps, device=text_projection.device).expand(bs, -1)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    

class Wav2Vec2ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=3):
        super(Wav2Vec2ContrastiveLoss, self).__init__()
        self.temp = temperature
        self.K = num_negatives
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, encoder_out, quantized_features, time_mask):
        """
        Args:
            encoder_out: [bs, T, dim]
            quantized_features: [bs, T, dim]
            time_mask: [bs, T]
        """
        batch_size, T, dim = encoder_out.shape

        # Masked encoder output (keep shape, zero out unmasked positions)
        masked_encoder_out = encoder_out * time_mask.unsqueeze(-1)  # (batch, seq_len, dim)
        masked_quantized_features = quantized_features * time_mask.unsqueeze(-1)  # (batch, seq_len, dim)

        # Sample negatives (batch, seq_len, K, dim)
        negative_samples, neg_mask = self.negative_sampler(quantized_features, time_mask)

        # Concatenate positive and negative samples (batch, seq_len, K+1, dim)
        all_samples = torch.cat([masked_quantized_features.unsqueeze(2), negative_samples], dim=2)
        # breakpoint()
        time_mask = time_mask.float() * neg_mask

        return self.contrastive_loss(masked_encoder_out, masked_quantized_features, negative_samples, time_mask)
    
    def contrastive_loss(self, targets, labels, negative_samples, time_mask_indices):
        """
        Computes contrastive loss.

        Args:
            targets (torch.Tensor): (batch, seq_len, dim)
            labels (torch.Tensor): (batch, seq_len, dim) - positive samples
            negative_samples (torch.Tensor): (batch, seq_len, K, dim) - negative samples
            time_mask_indices (torch.Tensor): (batch, seq_len) mask

        Returns:
            torch.Tensor: Scalar loss
        """
        batch_size, seq_len, dim = targets.shape
        K = negative_samples.shape[2]

        # # Normalize
        targets = F.normalize(targets, p=2, dim=-1)
        labels = F.normalize(labels, p=2, dim=-1)
        negative_samples = F.normalize(negative_samples, p=2, dim=-1)

        # Compute cosine similarities
        pos_similarity = self.cos(targets, labels) / self.temp  # (batch, seq_len)
        neg_similarity = self.cos(targets.unsqueeze(2), negative_samples) / self.temp  # (batch, seq_len, K)

        # Numerical stability: subtract max similarity before applying exp
        max_sim = torch.cat([pos_similarity.unsqueeze(-1), neg_similarity], dim=-1).max(dim=-1, keepdim=True).values
        pos_similarity = pos_similarity - max_sim.squeeze(-1)
        neg_similarity = neg_similarity - max_sim

        # Compute exp(similarity)
        exp_pos_sim = torch.exp(pos_similarity)  # (batch, seq_len)
        exp_neg_sim = torch.exp(neg_similarity).sum(dim=-1)  # (batch, seq_len)

        # Contrastive loss
        loss = -torch.log(exp_pos_sim / (exp_pos_sim + exp_neg_sim))

        # Zero out unmasked positions before averaging
        loss = loss * time_mask_indices  # (batch, seq_len)
        loss = loss.sum() / time_mask_indices.sum()  # Normalize over masked positions

        return loss
    
    def negative_sampler(self, quantized_features, time_mask_indices):
        """
        Samples K negative examples for each time step while ensuring they are not from masked positions.

        Args:
            quantized_features (torch.Tensor): (batch, seq_len, dim)
            time_mask_indices (torch.Tensor): Boolean mask (batch, seq_len)

        Returns:
            torch.Tensor: Negative samples (batch, seq_len, K, dim)
        """
        batch_size, seq_len, dim = quantized_features.shape
        negative_samples = torch.zeros((batch_size, seq_len, self.K, dim), device=quantized_features.device)
        neg_mask = torch.ones((batch_size, seq_len), device=quantized_features.device)

        for i in range(batch_size):
            # Get valid (unmasked) indices for sampling negatives
            valid_indices = (~time_mask_indices[i]).nonzero(as_tuple=True)[0]  # (num_valid_positions,)

            if len(valid_indices) == 0:
                # breakpoint()
                neg_mask[i] = 0
                continue

            if len(valid_indices) < self.K:
                # breakpoint()
                # raise ValueError(f"Number of valid positions ({len(valid_indices)}) is less than K ({self.K})")
                while len(valid_indices) < self.K:
                    valid_indices = torch.cat([valid_indices, valid_indices], dim=0)
                valid_indices = valid_indices[:self.K]
            
            # Sample K negatives for each time step
            for j in range(seq_len):
                # Sample K indices without replacement
                neg_indices = torch.randperm(len(valid_indices), device=quantized_features.device)[:self.K]

                # Gather negative samples
                negative_samples[i, j] = quantized_features[i, valid_indices[neg_indices]]

        # print(neg_mask)
        return negative_samples, neg_mask
    

class VILContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=3):
        super(VILContrastiveLoss, self).__init__()
        self.temp = temperature
        # self.K = num_negatives
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, input_seq, target, negative_samples, token_timemask):
        """
        Args:
            input_seq: [bs, T, dim]
            target: [bs, T, dim]
            negative_samples: [bs, T, K, dim]
            token_timemask: [bs, T]
        """
        return self.contrastive_loss(input_seq, target, negative_samples, token_timemask)

    def contrastive_loss(self, targets, labels, negative_samples, token_timemask):
        """
        Computes contrastive loss.

        Args:
            targets (torch.Tensor): (batch, seq_len, dim)
            labels (torch.Tensor): (batch, seq_len, dim) - positive samples
            negative_samples (torch.Tensor): (batch, seq_len, K, dim) - negative samples
            token_timemask (torch.Tensor): (batch, seq_len) mask for loss

        Returns:
            torch.Tensor: Scalar loss
        """
        batch_size, seq_len, dim = targets.shape
        K = negative_samples.shape[2]

        # breakpoint()

        # # Normalize
        targets = F.normalize(targets, p=2, dim=-1)
        labels = F.normalize(labels, p=2, dim=-1)
        negative_samples = F.normalize(negative_samples, p=2, dim=-1)

        # breakpoint()

        # Compute cosine similarities
        pos_similarity = self.cos(targets, labels) / self.temp  # (batch, seq_len)
        neg_similarity = self.cos(targets.unsqueeze(2), negative_samples) / self.temp  # (batch, seq_len, K)

        # Numerical stability: subtract max similarity before applying exp
        max_sim = torch.cat([pos_similarity.unsqueeze(-1), neg_similarity], dim=-1).max(dim=-1, keepdim=True).values
        pos_similarity = pos_similarity - max_sim.squeeze(-1)
        neg_similarity = neg_similarity - max_sim

        # Compute exp(similarity)
        exp_pos_sim = torch.exp(pos_similarity)  # (batch, seq_len)
        exp_neg_sim = torch.exp(neg_similarity).sum(dim=-1)  # (batch, seq_len)

        # Contrastive loss
        loss = -torch.log(exp_pos_sim / (exp_pos_sim + exp_neg_sim))

        # Zero out unmasked positions before averaging
        loss = loss * token_timemask  # (batch, seq_len)
        loss = loss.sum() / token_timemask.sum()  # Normalize over masked positions

        return loss



class TextLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=3, ignore_index=-100):
        super(TextLabelContrastiveLoss, self).__init__()
        self.temp = temperature
        self.K = num_negatives
        self.cos = nn.CosineSimilarity(dim=-1)
        self.ignore_index = ignore_index

    def forward(self, encoder_out, token_vecs, token_labels):
        """
        Args:
            encoder_out: [bs, T, dim]
            token_vecs: [bs, T, dim]
            token_labels: [bs, T]
        """
        # breakpoint()
        batch_size, T, dim = encoder_out.shape

        # # Masked encoder output (keep shape, zero out unmasked positions)
        # masked_encoder_out = encoder_out * time_mask.unsqueeze(-1)  # (batch, seq_len, dim)
        # masked_quantized_features = quantized_features * time_mask.unsqueeze(-1)  # (batch, seq_len, dim)

        token_timemask = (token_labels != self.ignore_index).float()
        # token_timemask = torch.ones_like(token_timemask)

        # Sample negatives (batch, seq_len, K, dim)
        negative_samples, neg_mask = self.negative_sampler(token_vecs, token_labels)

        token_timemask = token_timemask * neg_mask

        return self.contrastive_loss(encoder_out, token_vecs, negative_samples, token_timemask)
    
    def contrastive_loss(self, targets, labels, negative_samples, token_timemask):
        """
        Computes contrastive loss.

        Args:
            targets (torch.Tensor): (batch, seq_len, dim)
            labels (torch.Tensor): (batch, seq_len, dim) - positive samples
            negative_samples (torch.Tensor): (batch, seq_len, K, dim) - negative samples
            token_timemask (torch.Tensor): (batch, seq_len) mask for loss

        Returns:
            torch.Tensor: Scalar loss
        """
        batch_size, seq_len, dim = targets.shape
        K = negative_samples.shape[2]

        # # Normalize
        targets = F.normalize(targets, p=2, dim=-1)
        labels = F.normalize(labels, p=2, dim=-1)
        negative_samples = F.normalize(negative_samples, p=2, dim=-1)

        # breakpoint()

        # Compute cosine similarities
        pos_similarity = self.cos(targets, labels) / self.temp  # (batch, seq_len)
        neg_similarity = self.cos(targets.unsqueeze(2), negative_samples) / self.temp  # (batch, seq_len, K)

        # Numerical stability: subtract max similarity before applying exp
        max_sim = torch.cat([pos_similarity.unsqueeze(-1), neg_similarity], dim=-1).max(dim=-1, keepdim=True).values
        pos_similarity = pos_similarity - max_sim.squeeze(-1)
        neg_similarity = neg_similarity - max_sim

        # Compute exp(similarity)
        exp_pos_sim = torch.exp(pos_similarity)  # (batch, seq_len)
        exp_neg_sim = torch.exp(neg_similarity).sum(dim=-1)  # (batch, seq_len)

        # Contrastive loss
        loss = -torch.log(exp_pos_sim / (exp_pos_sim + exp_neg_sim))

        # Zero out unmasked positions before averaging
        loss = loss * token_timemask  # (batch, seq_len)
        loss = loss.sum() / token_timemask.sum()  # Normalize over masked positions

        return loss
    
    def negative_sampler(self, feature_vectors, feature_labels):
        """
        Samples K negative examples for each time step so that they are not from the same class.

        Args:
            feature_vectors (torch.Tensor): (batch, seq_len, dim)
            feature_labels (torch.Tensor): (batch, seq_len)

        Returns:
            torch.Tensor: Negative samples (batch, seq_len, K, dim)
        """
        batch_size, seq_len, dim = feature_vectors.shape
        negative_samples = torch.zeros((batch_size, seq_len, self.K, dim), device=feature_vectors.device)
        neg_mask = torch.ones((batch_size, seq_len), device=feature_vectors.device)

        # breakpoint()
        for i in range(batch_size):
            for j in range(seq_len):
                # Get valid (unmasked) indices for sampling negatives
                valid_indices = (feature_labels[i] != feature_labels[i, j]).nonzero(as_tuple=True)[0]
                # breakpoint()
                if len(valid_indices) == 0:
                    # breakpoint()
                    neg_mask[i, j] = 0
                    continue

                if len(valid_indices) < self.K:
                    # todo: handle this case properly such that if there is no negative sample, we can still compute the loss
                    # raise ValueError(f"Number of valid positions ({len(valid_indices)}) is less than K ({self.K})")
                    while len(valid_indices) < self.K:
                        valid_indices = torch.cat([valid_indices, valid_indices], dim=0)
                    valid_indices = valid_indices[:self.K]
                

                # Sample K negatives for each time step
                # Sample K indices without replacement
                neg_indices = torch.randperm(len(valid_indices), device=feature_vectors.device)[:self.K]

                # Gather negative samples
                negative_samples[i, j] = feature_vectors[i, valid_indices[neg_indices]]

        return negative_samples, neg_mask
    

class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Loss function to maximize cosine similarity between two embedding sequences.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, emb1, emb2):
        """
        Compute the loss to maximize cosine similarity.
        
        Args:
            emb1 (torch.Tensor): First embedding tensor of shape (batch_size, time, embedding_dim).
            emb2 (torch.Tensor): Second embedding tensor of shape (batch_size, time, embedding_dim).
        
        Returns:
            torch.Tensor: Loss value.
        """
        cosine_sim = F.cosine_similarity(emb1, emb2, dim=-1)  # Shape: (batch_size, time)
        loss = 1 - cosine_sim  # Maximizing similarity by minimizing (1 - cosine similarity)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction
        

class LaplacianLoss(nn.Module):
    def __init__(self, kernel_size=3, reduction='mean'):
        """
        kernel_size: size of Laplacian kernel (3, 5, 7, ...)
        """
        super().__init__()
        assert kernel_size in [3, 5, 7], "Supported kernel sizes: 3, 5, 7"
        self.kernel_size = kernel_size
        self.register_buffer("kernel", self._make_kernel(kernel_size))
        self.reduction = reduction

    def _make_kernel(self, k):
        if k == 3:
            coeffs = [1., -2., 1.]
        elif k == 5:
            coeffs = [-1., 16., -30., 16., -1.]
            coeffs = [c / 12. for c in coeffs]
        elif k == 7:
            coeffs = [2., -27., 270., -490., 270., -27., 2.]
            coeffs = [c / 180. for c in coeffs]
        kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, k)
        return kernel

    def forward(self, x_hat, x):
        """
        x_hat: predicted motion sequence, shape (B, T, C)
        x: ground truth motion sequence, shape (B, T, C)
        """
        # Rearrange to (B, C, T) for conv1d
        x_hat = x_hat.permute(0, 2, 1)
        x = x.permute(0, 2, 1)

        # Apply Laplacian filter along temporal dimension
        kernel = self.kernel.expand(x.size(1), -1, -1).to(x.device)  # Shape: (C, 1, k)
        lap_x_hat = F.conv1d(x_hat, kernel, padding=self.kernel_size // 2, groups=x.size(1))
        lap_x = F.conv1d(x, kernel, padding=self.kernel_size // 2, groups=x.size(1))

        lap_x_hat = lap_x_hat.permute(0, 2, 1)  # Back to (B, T, C)
        lap_x = lap_x.permute(0, 2, 1)          # Back to (B, T, C)

        if self.reduction == 'mean':
            return ((lap_x_hat - lap_x) ** 2).mean()
        else:
            return (lap_x_hat - lap_x) ** 2

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss for aligning two distributions.
    Can be used to align latent spaces from two datasets (e.g., dataset A and B).
    """

    def __init__(self, kernel_type='rbf', sigma_list=None, normalize=True):
        """
        Args:
            kernel_type (str): 'rbf' (Gaussian) or 'linear'.
            sigma_list (list): list of kernel bandwidths for RBF. 
                               If None, defaults to [0.1, 1.0, 5.0, 10.0].
        """
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        if sigma_list is None:
            sigma_list = [0.1, 1.0, 5.0, 10.0]
        self.sigma_list = sigma_list
        self.normalize = normalize

    def gaussian_kernel(self, x, y):
        """
        Compute RBF kernel between x and y with multiple bandwidths.
        """
        if self.normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        
        # x: [n, d], y: [m, d]
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)   # [n,1]
        y_norm = (y ** 2).sum(dim=1).view(1, -1)   # [1,m]
        dist2 = x_norm + y_norm - 2 * torch.mm(x, y.t())
        dist2 = torch.clamp(dist2, min=0, max=1e6)

        kernels = [torch.exp(-dist2 / (2 * sigma ** 2)) for sigma in self.sigma_list]
        return sum(kernels) / len(kernels)

    def forward(self, x, y):
        """
        Args:
            x (Tensor): [n, d] latent samples from dataset A
            y (Tensor): [m, d] latent samples from dataset B
        Returns:
            Scalar tensor: MMD loss
        """
        if self.kernel_type == 'linear':
            # Linear kernel MMD
            xx = torch.mean(torch.mm(x, x.t()))
            yy = torch.mean(torch.mm(y, y.t()))
            xy = torch.mean(torch.mm(x, y.t()))
            return xx + yy - 2 * xy

        elif self.kernel_type == 'rbf':
            # breakpoint()
            Kxx = self.gaussian_kernel(x, x)
            Kyy = self.gaussian_kernel(y, y)
            Kxy = self.gaussian_kernel(x, y)
            return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

        else:
            raise ValueError("Unsupported kernel type: {}".format(self.kernel_type))


class SmoothnessLoss(nn.Module):
    """
    Smoothness loss combining velocity continuity and acceleration penalties.
    Options for reduction: 'mean', 'huber', 'topk'.
    """

    def __init__(self, lambda_vel=1e-2, lambda_acc=1e-3,
                 reduction='mean', huber_delta=1.0, topk_ratio=0.1):
        """
        Args:
            lambda_vel (float): weight for velocity continuity loss
            lambda_acc (float): weight for acceleration loss
            reduction (str): 'mean', 'huber', or 'topk'
            huber_delta (float): delta for Huber loss if reduction='huber'
            topk_ratio (float): fraction of timesteps to average if reduction='topk'
        """
        super(SmoothnessLoss, self).__init__()
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc #* 0.1
        self.reduction = reduction
        self.huber_delta = huber_delta
        self.topk_ratio = topk_ratio

    def _apply_reduction(self, values):
        """Apply reduction method to [B, T] tensor of per-frame penalties."""
        if self.reduction == 'mean':
            return values.mean()

        elif self.reduction == 'huber':
            return F.huber_loss(values, torch.zeros_like(values),
                                delta=self.huber_delta, reduction='mean')

        elif self.reduction == 'topk':
            k = max(1, int(self.topk_ratio * values.shape[1]))
            topk_vals, _ = torch.topk(values, k, dim=1)
            return topk_vals.mean()

        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

    def forward(self, x_recon):
        """
        Args:
            x_recon: [B, T, D] reconstructed sequence
        Returns:
            scalar smoothness loss
        """
        B, T, D = x_recon.shape

        # velocity: v_t = x_{t+1} - x_t
        vel = x_recon[:, 1:, :] - x_recon[:, :-1, :]      # [B, T-1, D]
        vel_diff = vel[:, 1:, :] - vel[:, :-1, :]         # [B, T-2, D]
        vel_penalty = (vel_diff ** 2).sum(dim=-1)         # [B, T-2]

        # acceleration: a_t = x_{t+2} - 2x_{t+1} + x_t
        accel = x_recon[:, 2:, :] - 2 * x_recon[:, 1:-1, :] + x_recon[:, :-2, :]
        accel_diff = accel[:, 1:, :] - accel[:, :-1, :]     # [B, T-3, D]
        acc_penalty = (accel_diff ** 2).sum(dim=-1)            # [B, T-3]

        loss_vel = self._apply_reduction(vel_penalty)
        loss_acc = self._apply_reduction(acc_penalty)

        return self.lambda_vel * loss_vel + self.lambda_acc * loss_acc

class ContrastiveMMDLatentLoss(nn.Module):
    def __init__(self, temperature=0.1, segment_len=4):
        super(ContrastiveMMDLatentLoss, self).__init__()
        self.temperature = temperature
        self.segment_len = segment_len # number of latent frames to average over
        
    def compute_contrastive_loss(self, z_real, z_fake):
        tau = self.temperature
        segment_len = self.segment_len

        # breakpoint()

        B, T, D = z_real.shape
        num_segments = T // segment_len
        z_real = z_real[:, :num_segments*segment_len].reshape(B, num_segments, segment_len, D).mean(dim=2)
        z_fake = z_fake[:, :num_segments*segment_len].reshape(B, num_segments, segment_len, D).mean(dim=2)
        
        z_real = F.normalize(z_real.reshape(B*num_segments, D), dim=-1)
        z_fake = F.normalize(z_fake.reshape(B*num_segments, D), dim=-1)
        
        sim = torch.matmul(z_real, z_fake.T) / tau
        labels = torch.arange(len(z_real), device=z_real.device)
        loss = F.cross_entropy(sim, labels)
        return loss
    
    def compute_mmd(self, z_real: torch.Tensor, z_fake: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) between two latent distributions.
        Numerically stable version.
        """

        B, T, D = z_real.shape
        z_real = z_real.reshape(B * T, D)
        z_fake = z_fake.reshape(B * T, D)

        # Normalize to prevent exploding norms (optional but helps)
        z_real = z_real - z_real.mean(0, keepdim=True)
        z_fake = z_fake - z_fake.mean(0, keepdim=True)

        def pdist(x):
            """Compute pairwise squared distance with numerical stability."""
            x_norm = (x ** 2).sum(dim=1, keepdim=True)
            dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())
            dist = torch.clamp(dist, min=0.0)  # avoid tiny negatives due to float error
            return dist

        # Add a small jitter to diagonal for numerical stability
        eps = 1e-6
        K_xx = torch.exp(-pdist(z_real) / (2 * sigma ** 2))
        K_yy = torch.exp(-pdist(z_fake) / (2 * sigma ** 2))
        K_xy = torch.exp(
            -torch.clamp(
                ( (z_real ** 2).sum(1, keepdim=True)
                + (z_fake ** 2).sum(1).unsqueeze(0)
                - 2 * (z_real @ z_fake.t())), min=0.0
            ) / (2 * sigma ** 2)
        )

        # Zero out diagonals to remove self-similarity bias
        diag_mask_xx = 1 - torch.eye(K_xx.size(0), device=z_real.device)
        diag_mask_yy = 1 - torch.eye(K_yy.size(0), device=z_fake.device)
        K_xx = K_xx * diag_mask_xx + eps
        K_yy = K_yy * diag_mask_yy + eps

        # Compute unbiased MMD estimate
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd


def stopgrad(x):
    return x.detach()

class AdaptiveL2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AdaptiveL2Loss, self).__init__()
        self.reduction = reduction
    
    def forward(self, outputs, targets, gamma=0.5, c=1e-3):
        """
        Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        outputs: predicted values, shape (B, T, P, D)
        targets: ground truth values, shape (B, T, P, D)
        """
        error = outputs - targets  # (B, T, P, D)
        error_squared = (error ** 2).mean(dim=-1)  # (B, T, P)
        p = 1.0 - gamma
        weights = 1.0 / (error_squared + c).pow(p)  # (B, T, P)
        loss = error_squared
        loss = stopgrad(weights) * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {self.reduction}')
    

class RVQ_ClassificationLoss(nn.Module):
    def __init__(self, codebooks, lambda_per_level=None, reduction='mean'):
        super(RVQ_ClassificationLoss, self).__init__()
        if lambda_per_level is None:
            lambda_per_level = [1.0] * len(codebooks)

        self.reduction = reduction

        self.codebooks = codebooks
        self.lambda_per_level = lambda_per_level

    def forward(self, z_pred, gt_indices):
        """
        Args:
            z_pred: predicted latent vectors, shape (B, T, D)
            gt_indices: ground-truth codebook indices, shape (B, K, T) where K is number of codebooks
        """
        residual = z_pred
        total_loss = 0.0
        K = len(self.codebooks)
        assert K == len(self.lambda_per_level), "Length of lambda_per_level must match number of codebooks"
        assert K == gt_indices.shape[1], "Number of codebooks must match number of levels in gt_indices"

        for k in range(K):
            # breakpoint()
            codebook_k = self.codebooks[k]  # (num_codes, code_dim)
            gt_idx_k = gt_indices[:, k]     # (B, T)
            lam = self.lambda_per_level[k]
            
            
            distances = torch.cdist(residual, codebook_k.unsqueeze(0), p=2)  # (..., K_l)
            logits = -distances ** 2

            # Cross-entropy between logits and ground-truth index
            loss_l = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                gt_idx_k.view(-1),
                reduction=self.reduction
            )
            total_loss += lam * loss_l

            # Compute predicted embedding for next residual
            with torch.no_grad():  # no gradient through quantization
                nearest = distances.argmin(dim=-1)  # (...,)
                e_q = codebook_k[nearest]             # (..., D)
                residual = residual - e_q           # update residual
        
        total_loss = total_loss / K  # average over codebooks
        if self.reduction == "none":
            total_loss = total_loss.view(z_pred.size(0), z_pred.size(1))  # (B, T)
        
        return total_loss




LOSS_FUNC_LUT = {
        "bce_loss": BCE_Loss,
        "l2_loss": L2Loss,
        "huber_loss": HuberLoss,
        "KLLoss": KLLoss,
        "KLLossMulti": KLLossMulti,
        "id_loss": REGLoss,
        "GeodesicLoss": GeodesicLoss,
        "weight_Loss": weight_Loss,

        "info_nce": ContrastiveLoss,
        "wav2vec2_contrastive": Wav2Vec2ContrastiveLoss,
        "text_contrastive": TextLabelContrastiveLoss,
        "CosineSimilarityLoss": CosineSimilarityLoss,
        "vil_contrastive": VILContrastiveLoss,
        "laplacian_loss": LaplacianLoss,
        "mmd_loss": MMDLoss,
        "smoothness_loss": SmoothnessLoss,
        "contrastive_latent_loss": ContrastiveMMDLatentLoss,

        "adaptive_l2_loss": AdaptiveL2Loss,

        "rvq_index_loss": RVQ_ClassificationLoss,
    }


def get_loss_func(loss_name, **kwargs):    
    loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    loss_func = loss_func_class(**kwargs)   
    return loss_func