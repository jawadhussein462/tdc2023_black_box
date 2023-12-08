import torch 
import torch.nn as nn
import torch.nn.functional as F

class SurrogateModel(nn.Module):
    """
    A surrogate model for embedding-based optimization tasks. 

    This model is designed to work with a set of precomputed embeddings (ref_emb) and predicts an output value based on a combination of these embeddings. The model consists of a series of fully connected layers and batch normalization layers.

    Attributes:
    emb_dim (int): Dimension of the embeddings.
    len_coordinates (int): The number of coordinates (or embeddings) used in each input.
    emb (torch.Tensor): A tensor containing the reference embeddings.
    fc1, fc2, fc3 (nn.Linear): Fully connected layers of the model.
    bn1, bn2 (nn.BatchNorm1d): Batch normalization layers.

    Methods:
    forward(str_id): Performs a forward pass of the model. The input str_id is used to select embeddings from the reference set, which are then processed through the model's layers to produce an output.
    """
    
    def __init__(self, len_coordinates, ref_emb):
        super(SurrogateModel, self).__init__()

        self.emb_dim = ref_emb.shape[1]
        self.len_coordinates = len_coordinates
        self.emb = ref_emb.clone()
        self.emb.require_grad = False

        self.fc1 = nn.Linear(self.emb_dim*self.len_coordinates, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, str_id):

        str_emb = self.emb[str_id]
        x = torch.flatten(str_emb, start_dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class AcquisitionFunction(nn.Module):
    """
    An acquisition function module for optimization in a surrogate model.

    This module is designed to facilitate the selection of the next set of points to evaluate in the context of optimization problems. 
    It uses a surrogate model to predict values for a range of inputs and selects the top candidates based on these predictions.

    Attributes:
    max_dim (int): The maximum dimension of the input space.
    len_coordinates (int): The number of coordinates in each input.
    indices (torch.Tensor): A tensor of indices representing the input space.
    
    Methods:
    forward(surrogate_model, str_id, best_score, coordinate, num_samples, str_ids_ignore): Performs a forward pass. 
    This function generates a set of inputs, predicts their outcomes using the surrogate model, and selects the top candidates based on these predictions.
    compute_mask(str_ids_ignore, str_id, coordinate): A helper function to compute a mask for filtering out certain indices during the forward pass.
    """    
    
    def __init__(self, max_dim, len_coordinates, device):
        super(AcquisitionFunction, self).__init__()
        self.max_dim = max_dim
        self.len_coordinates = len_coordinates
        self.indices = torch.arange(0, max_dim).long().to(device)

    def forward(self,
                surrogate_model,
                str_id,
                coordinate,
                num_samples,
                str_ids_ignore=None):

        inputs = str_id.repeat(self.max_dim, 1)
        inputs[:, coordinate] = self.indices

        predictions = surrogate_model(inputs).T
        adjusted_predictions = predictions.clone()


        with torch.no_grad():

            if str_ids_ignore:
                str_ids_ignore_concat = torch.cat(str_ids_ignore)
                mask = self.compute_mask(str_ids_ignore_concat, str_id, coordinate)

                if mask.sum() > 0:
                    count = torch.nn.functional.one_hot(str_ids_ignore_concat[mask, coordinate], num_classes=self.max_dim).sum(0)
                    adjusted_predictions = adjusted_predictions - 10000000 * count

            top_indices = torch.topk(adjusted_predictions, num_samples).indices.view(-1)
            self.top_indices = top_indices

        self.predictions = predictions
        values = predictions[:, top_indices]
        top_inputs = inputs[top_indices, :]

        return top_inputs, values

    def compute_mask(self, str_ids_ignore, str_id, coordinate):
        d1 = torch.eq(str_ids_ignore[:, :coordinate], str_id[:, :coordinate]).all(dim=-1)
        d2 = torch.eq(str_ids_ignore[:, coordinate+1:], str_id[:, coordinate+1:]).all(dim=-1)

        if coordinate == 0:
            mask = d2
        elif coordinate == self.len_coordinates - 1:
            mask = d1
        else:
            mask = d1 & d2

        return mask