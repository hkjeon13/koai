from typing import Any, Tuple, Union
from transformers import Trainer
import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        """
        Compute contrastive loss

        Args:
            output1: One Vector (batch_size, embedding_size)
            output2: The other Vector (batch_size, embedding_size)

        Returns:
            loss_contrastive: Contrastive loss

        >>> import torch
        >>> output1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> output2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> loss = ContrastiveLoss()
        >>> result = loss(output1, output2)
        >>> str(result)
        'tensor(1.0000)'
        """
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - euclidean_distance) ** 2)
        return loss_contrastive


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        """
        Trainer For the Contrastive Loss
        Args:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.contrastive_loss = ContrastiveLoss()

    def compute_loss(self, model, inputs, return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute the loss for the model
        Args:
            model:  The model 
            inputs: The inputs
            return_outputs: Return the outputs 
        
        Returns:
            loss: The loss

        """
        outputs = model(**inputs)
        loss = self.contrastive_loss(outputs[0], outputs[1])
        return (loss, outputs) if return_outputs else loss