import torch 
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, values, score):

      # Sum the result
      loss = (values - score)**2

      return loss.mean()


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, values, score):
        # Ensure values and score are 1D tensors
        assert len(values.shape) == 1 and len(score.shape) == 1

        # Compute all pairwise differences for the predicted scores and true scores
        diff_values = values.unsqueeze(0) - values.unsqueeze(1)
        diff_score = score.unsqueeze(0) - score.unsqueeze(1)

        # Mask where true score of item i > item j
        mask = (diff_score > 0).float()*2-1

        # Calculate hinge loss for each pair
        pairwise_losses = F.relu(self.margin - mask * diff_values)

        return pairwise_losses.mean()