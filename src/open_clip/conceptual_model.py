import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sparsemax import Sparsemax
from typing import Tuple, Optional, Any, Dict
from dataclasses import dataclass, asdict


from .modified_resnet import ModifiedResNet
from .model import CustomTextCLIP, CLIPTextCfg, CLIPVisionCfg


@dataclass
class CLIPConceptualCfg:
    conceptual_type: str = None
    num_low_concepts: int = None
    norm_low_concepts: bool = False
    num_high_concepts: int = None
    norm_high_concepts: bool = False
    low_high_max_function: str = None
    output_high_concepts_type: str = None
    detach_low_concepts: bool = False
    learnable_hierarchy: bool = False
    preset_hierarchy: bool = False
    spatial_dim: int = 7
    image_high_concept_n_head: int = None
    image_high_concept_keep_head_dim: bool = False
    image_high_concept_max_function_name: str = None
    image_high_concept_max_smoothing: float = 0.0
    image_patch_n_head: int = None
    image_patch_keep_head_dim: bool = False
    image_patch_max_function_name: str = None
    image_patch_max_smoothing: float = 0.0
    patch_low_concept_n_head: int = None
    patch_low_concept_keep_head_dim: bool = False
    patch_low_concept_max_function_name: str = None
    patch_low_concept_max_smoothing: float = 0.0


class ModifiedResNetRemovedAttnPool(ModifiedResNet):
    def __init__(self, layers, image_size=224, width=64):
        super().__init__(
            layers=layers,
            output_dim=None,
            heads=None,
            image_size=image_size,
            width=width,
        )
        self.attnpool = None
        self.init_parameters()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_max_function(max_function_name: str, dim: int = -1) -> Any:
    """
    Returns a max function given its name and optional arguments.

    Args:
        max_function_name (str): The name of the max function to return.
        dim (int): The dimension along which to apply the max function.

    Returns:
        Any: The max function.

    Raises:
        ValueError: If an invalid max function name is provided.
    """
    max_functions = {
        "softmax": nn.Softmax(dim=dim),
        "hardmax": Hardmax(dim=dim),
        "sparsemax": Sparsemax(dim=dim),
        "gumbel": GumbelSoftmax(dim=dim),
        "hard_gumbel": GumbelSoftmax(hard=True, dim=dim)
    }

    if max_function_name in max_functions:
        return max_functions[max_function_name]
    else:
        raise ValueError(f"Invalid max function: {max_function_name}")


class Hardmax(nn.Module):
    """
    A hardmax function that returns a one-hot tensor.

    Args:
        dim (int): The dimension along which to apply the hardmax.

    Returns:
        Tensor: The one-hot tensor.
    """

    def __init__(self, dim: int = -1):
        super(Hardmax, self).__init__()
        self.dim = dim

    def forward(self, logits):
        max_index = torch.argmax(logits, dim=self.dim)
        return F.one_hot(max_index, num_classes=logits.size(self.dim)).float()


class GumbelSoftmax(nn.Module):
    """
    Gumbel Softmax module.

    Args:
        hard (bool, optional): Whether to use hard Gumbel softmax (default: False).
        dim (int, optional): The dimension along which the Gumbel softmax is applied (default: -1).
        eps (float, optional): A small value added to the denominator for numerical stability (default: 1e-10).

    Attributes:
        temperature (torch.Tensor): A learnable parameter that controls the temperature of the Gumbel-Softmax function.
        tau (float): A constant that controls the behavior of the Gumbel-Softmax function.
        hard (bool): If True, uses hard Gumbel-Softmax for discrete sampling.
        dim (int): The dimension along which to apply the Gumbel-Softmax function.
        eps (float): A small value to ensure numerical stability.

    Examples:
        >>> gs = GumbelSoftmax()
        >>> x = torch.randn(2, 10)
        >>> y = gs(x)
    """

    def __init__(self, hard=False, dim=-1, eps=1e-10):
        super(GumbelSoftmax, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(0.0))
        self.tau = 1.0
        self.hard = hard
        self.dim = dim
        self.eps = eps

    def forward(self, logits):
        if self.training:
            return F.gumbel_softmax(logits * self.temperature.exp(), tau=self.tau, hard=self.hard, dim=self.dim, eps=self.eps)
        else:
            index = logits.max(self.dim, keepdim=True)[1]
            logits_hard = torch.zeros_like(
                logits,
                memory_format=torch.legacy_contiguous_format
            ).scatter_(self.dim, index, 1.0)
        return logits_hard


class ConfigurableMultiHeadAttention(nn.Module):
    """
    A PyTorch module for implementing configurable multi-head attention mechanism.

    Args:
        query_dim (int): The dimension of the query vectors.
        key_dim (int): The dimension of the key vectors.
        n_head (int): The number of attention heads.
        keep_head_dim (bool): Whether to keep the head dimension or not.
        max_function_name (str): The name of the maximum function to use.
        max_smoothing (float, optional): The smoothing factor for the maximum function. Defaults to 0.0.
        learnable (bool, optional): Whether the attention mechanism is learnable or not. Defaults to True.
    """

    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 n_head: int,
                 keep_head_dim: bool,
                 max_function_name: str,
                 max_smoothing: float = 0.0,
                 learnable: bool = True) -> None:
        super(ConfigurableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        assert max_smoothing >= 0.0 and max_smoothing <= 1.0
        self.max_smoothing = max_smoothing
        self.learnable = learnable

        if self.learnable:

            if keep_head_dim:
                self.d_head = key_dim
                self.q_linear = nn.Linear(
                    query_dim, key_dim*n_head, bias=False
                )
                self.k_linear = nn.Linear(
                    key_dim, key_dim*n_head, bias=False
                )
            else:
                assert key_dim % n_head == 0
                self.d_head = key_dim // n_head
                self.q_linear = nn.Linear(query_dim, key_dim, bias=False)
                self.k_linear = nn.Linear(key_dim, key_dim, bias=False)

            nn.init.xavier_uniform_(self.q_linear.weight)
            nn.init.xavier_uniform_(self.k_linear.weight)

        else:
            assert self.n_head == 1
            assert query_dim == key_dim
            self.d_head = key_dim

        self.max_function = get_max_function(
            max_function_name=max_function_name,
            dim=-1
        )

    def smooth_max(self, max_output: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        A function that applies smoothing to the maximum output.

        Args:
            max_output (torch.Tensor): The maximum output tensor.
            dim (int, optional): The dimension to apply smoothing. Defaults to -1.

        Returns:
            torch.Tensor: The smoothed maximum output tensor.
        """
        if self.max_smoothing > 0.0 and self.training:
            max_output = max_output * (1.0 - self.max_smoothing) + \
                self.max_smoothing / max_output.size(dim)
        return max_output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method of the multi-head attention mechanism.

        Args:
            q (torch.Tensor): The query tensor (batch_size_q, seq_len_q, query_dim).
            k (torch.Tensor): The key tensor (batch_size_kv, seq_len_kv, key_dim).
            v (torch.Tensor): The value tensor (batch_size_kv, seq_len_kv, value_dim).
            k_mask (torch.Tensor, optional): The key mask tensor. Defaults to None (batch_size_q, seq_len_kv).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor (batch_size_q, seq_len_q, value_dim) 
                                               and the attention matrix (batch_size_q, seq_len_q, seq_len_kv).
        """
        if self.learnable:
            q = self.q_linear(q)
            k = self.k_linear(k)

        # [B_q, H, L_q, d_head]
        q = q.view(q.size(0), -1, self.n_head, self.d_head).transpose(1, 2)
        # [B_kv, H, L_kv, d_head]
        k = k.view(k.size(0), -1, self.n_head, self.d_head).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.d_head ** 0.5)  # [B_q, H, L_q, L_kv]

        if k_mask is not None:
            k_mask = k_mask.unsqueeze(1).unsqueeze(2)  # [B_q, 1, 1, L_kv]
            logits = logits * k_mask - (1.0 - k_mask) * 1e9

        attn = self.max_function(logits)  # [B_q, H, L_q, L_kv]

        if k_mask is not None:
            attn = attn * k_mask

        attn = attn.mean(dim=1)  # [B_q, L_q, L_kv]
        output = torch.matmul(self.smooth_max(attn), v)  # [B_q, L_q, D_v]

        return output, attn


class Concepts(nn.Module):
    def __init__(self, num_concepts: int, concept_dim: int):
        """
        Initializes a new instance of the `Concepts` class.

        Args:
            num_concepts (int): The number of concepts in the concept matrix.
            concept_dim (int): The dimension of each concept vector.
        """
        super(Concepts, self).__init__()

        self.concepts = nn.Parameter(
            torch.Tensor(num_concepts, concept_dim)
        )
        nn.init.xavier_uniform_(self.concepts)

    def forward(self, norm_concepts: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the `Concepts` module.

        Args:
            norm_concepts (bool): Whether to normalize the concept vectors or not.

        Returns:
            A tuple containing the concept matrix and the cosine similarity matrix between the normalized
            concept vectors.
        """
        normalized_concepts = self.concepts / \
            self.concepts.norm(dim=1, keepdim=True)
        # Return normalized concept vectors if requested.
        returned_concepts = normalized_concepts if norm_concepts else self.concepts

        # Compute the cosine similarity matrix between normalized concepts.
        concept_cosine_similarity = torch.matmul(
            normalized_concepts, normalized_concepts.t()
        )

        return returned_concepts, concept_cosine_similarity


def normalize_rows(input_tensor: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Normalize the rows of a tensor.

    Args:
        input_tensor: The input tensor to be normalized.
        epsilon: A small value added to the row sums to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    input_tensor = input_tensor.to(torch.float)
    row_sums = torch.sum(input_tensor, dim=1, keepdim=True)
    row_sums += epsilon
    normalized_tensor = input_tensor / row_sums
    return normalized_tensor


class HierarchicalConcepts(nn.Module):
    """
    A PyTorch module for implementing hierarchical concepts.

    Args:
        num_low_concepts (int): The number of low-level concepts.
        num_high_concepts (int): The number of high-level concepts.
        concept_dim (int): The dimension of each concept vector.
        low_high_max_function (str): The name of the maximum function to use for the low-high attention.
        output_high_concepts_type (str): The type of the output high-level concepts.
        learnable_hierarchy (bool, optional): Whether the hierarchy is learnable or not. Defaults to True.
    """

    def __init__(self,
                 num_low_concepts: int,
                 num_high_concepts: int,
                 concept_dim: int,
                 low_high_max_function: str,
                 output_high_concepts_type: str,
                 learnable_hierarchy: bool = True):
        super(HierarchicalConcepts, self).__init__()

        self.low_concepts = Concepts(
            num_concepts=num_low_concepts,
            concept_dim=concept_dim
        )
        self.high_concepts = Concepts(
            num_concepts=num_high_concepts,
            concept_dim=concept_dim
        )

        self.concept_hierarchy_builder = ConfigurableMultiHeadAttention(
            query_dim=concept_dim,
            key_dim=concept_dim,
            n_head=1,
            keep_head_dim=True,
            max_function_name=low_high_max_function,
            max_smoothing=0.0,
            learnable=learnable_hierarchy
        )
        self.output_high_concepts_type = output_high_concepts_type

    def forward(self, norm_low_concepts: bool, norm_high_concepts: bool, detach_low_concepts: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the hierarchical concepts module.

        Args:
            norm_low_concepts (bool): Whether to normalize the low-level concepts.
            norm_high_concepts (bool): Whether to normalize the high-level concepts.
            detach_low_concepts (bool): Whether to detach the low-level concepts related tensors.

        Returns:
            A tuple containing the following elements:
            - low_concepts (Tensor): The low-level concepts.
            - low_concept_cosine_similarity (Tensor): The cosine similarity between
              the low-level concepts and the input features.
            - output_high_concepts (Tensor): The output high-level concepts.
            - high_concept_cosine_similarity (Tensor): The cosine similarity between
              the high-level concepts and the input features.
            - low_high_hierarchy (Tensor): The hierarchy between low-level and
              high-level concepts.

        """
        low_concepts, low_concept_cosine_similarity = self.low_concepts(
            norm_low_concepts
        )
        high_concepts, high_concept_cosine_similarity = self.high_concepts(
            norm_high_concepts
        )

        if detach_low_concepts:
            high_related_low_concepts = low_concepts.detach()
        else:
            high_related_low_concepts = low_concepts

        _, low_high_hierarchy = self.concept_hierarchy_builder(
            high_related_low_concepts.unsqueeze(0),  # [1, N_low, D]
            high_concepts.unsqueeze(0),  # [1, N_high, D]
            high_concepts.unsqueeze(0)  # [1, N_high, D]
        )  # [1, N_low, N_high]

        low_high_hierarchy = low_high_hierarchy.squeeze(0)  # [N_low, N_high]

        if self.output_high_concepts_type == "original_high":
            output_high_concepts = high_concepts
        else:
            high_low_hierarchy = low_high_hierarchy.t()  # [N_high, N_low]
            output_high_concepts = torch.matmul(
                normalize_rows(high_low_hierarchy), high_related_low_concepts
            )  # [N_high, D]
            if self.output_high_concepts_type == "high_plus_low":
                output_high_concepts = output_high_concepts + high_concepts
            else:
                assert self.output_high_concepts_type == "aggregated_low"

        return (
            low_concepts,  # [N_low, D]
            low_concept_cosine_similarity,
            output_high_concepts,  # [N_high, D]
            high_concept_cosine_similarity,
            low_high_hierarchy  # [N_low, N_high]
        )


class Conceptualizer(nn.Module):
    """
    A PyTorch module for implementing the conceptualizer.

    Args:
        feature_dim (int): The dimension of the input features.
        concept_dim (int): The dimension of the concepts.
        n_head (int): The number of attention heads.
        keep_head_dim (bool): Whether to keep the head dimension or not.
        max_function_name (str): The name of the maximum function to use.
        max_smoothing (float, optional): The smoothing factor for the maximum function. Defaults to 0.0.
    """

    def __init__(self,
                 feature_dim: int,
                 concept_dim: int,
                 n_head: int,
                 keep_head_dim: bool,
                 max_function_name: str,
                 max_smoothing: float = 0.0):
        super(Conceptualizer, self).__init__()

        self.conceptual_attention = ConfigurableMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=concept_dim,
            n_head=n_head,
            keep_head_dim=keep_head_dim,
            max_function_name=max_function_name,
            max_smoothing=max_smoothing,
            learnable=True
        )

    def forward(self, x: torch.Tensor, concepts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # [B, 1, D_q]
        concepts = concepts.unsqueeze(0)  # [1, N, D_kv]
        conceptual_x, concept_attention_weight = self.conceptual_attention(
            x, concepts, concepts
        )  # [B, 1, D_kv], [B, 1, N]

        # [B, D_kv], [B, N]
        return conceptual_x.squeeze(1), concept_attention_weight.squeeze(1)


class MaskedConceptualPool2d(nn.Module):
    """
    A module that performs conceptual pooling on a 2D image tensor.

    Args:
        spatial_dim (int): The spatial dimension of the image.
        feature_dim (int): The dimension of the input features.
        concept_dim (int): The dimension of the concepts.
        image_patch_n_head (int): The number of attention heads for the image patch attention.
        image_patch_keep_head_dim (bool): Whether to keep the head dimension or not for the image patch attention.
        image_patch_max_function_name (str): The name of the maximum function to use for the image patch attention.
        image_patch_max_smoothing (float, optional): The smoothing factor for the maximum function for the image patch attention. Defaults to 0.0.
        patch_concept_n_head (int): The number of attention heads for the patch concept attention.
        patch_concept_keep_head_dim (bool): Whether to keep the head dimension or not for the patch concept attention.
        patch_concept_max_function_name (str): The name of the maximum function to use for the patch concept attention.
        patch_concept_max_smoothing (float, optional): The smoothing factor for the maximum function for the patch concept attention. Defaults to 0.0.
    """

    def __init__(self,
                 spatial_dim: int,
                 feature_dim: int,
                 concept_dim: int,
                 image_patch_n_head: int,
                 image_patch_keep_head_dim: bool,
                 image_patch_max_function_name: str,
                 image_patch_max_smoothing: float,
                 patch_concept_n_head: int,
                 patch_concept_keep_head_dim: bool,
                 patch_concept_max_function_name: str,
                 patch_concept_max_smoothing: float):
        super(MaskedConceptualPool2d, self).__init__()

        # positional embedding initialization
        self.positional_embedding = nn.Parameter(
            torch.randn(spatial_dim ** 2 + 1, feature_dim) / feature_dim ** 0.5
        )
        # image spatial attention
        self.image_patch_attention = ConfigurableMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=feature_dim,
            n_head=image_patch_n_head,
            keep_head_dim=image_patch_keep_head_dim,
            max_function_name=image_patch_max_function_name,
            max_smoothing=image_patch_max_smoothing,
            learnable=True
        )

        # patch conceptual attention
        self.patch_concept_attention = ConfigurableMultiHeadAttention(
            query_dim=feature_dim,
            key_dim=concept_dim,
            n_head=patch_concept_n_head,
            keep_head_dim=patch_concept_keep_head_dim,
            max_function_name=patch_concept_max_function_name,
            max_smoothing=patch_concept_max_smoothing,
            learnable=True
        )

    def forward(self, patches: torch.Tensor, concepts: torch.Tensor, concept_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # patch conceptual attention
        concepts = concepts.unsqueeze(0)  # [1, N, D_kv]
        conceptual_patches, patch_concept_attention_weight = self.patch_concept_attention(
            patches[:, 1:], concepts, concepts, concept_mask
        )  # [B, H*W, D_kv], [B, H*W, N]

        # image spatial attention
        # reshape positional_embedding [1+H*W, D_q] to [1, 1+H*W, D_q]
        positional_embedding = self.positional_embedding.unsqueeze(0)
        patches = patches + positional_embedding
        conceptual_image, image_patch_attention_weight = self.image_patch_attention(
            patches[:, :1], patches[:, 1:], conceptual_patches
        )  # [B, 1, D_kv], [B, 1, H*W]

        conceptual_image = conceptual_image.squeeze(1)  # [B, D_kv]
        image_concept_attention_weight = torch.matmul(
            image_patch_attention_weight, patch_concept_attention_weight
        ).squeeze(1)  # [B, N]
        image_patch_attention_weight = image_patch_attention_weight.squeeze(
            1)  # [B, H*W]

        return (
            conceptual_image,  # [B, D_kv]
            image_concept_attention_weight,  # [B, N]
            image_patch_attention_weight,  # [B, H*W]
            patch_concept_attention_weight  # [B, H*W, N]
        )


class TopDownHierConceptualPool2d(nn.Module):
    def __init__(self,
                 spatial_dim: int,
                 feature_dim: int,
                 concept_dim: int,
                 image_high_concept_n_head: int,
                 image_high_concept_keep_head_dim: bool,
                 image_high_concept_max_function_name: str,
                 image_high_concept_max_smoothing: float,
                 image_patch_n_head: int,
                 image_patch_keep_head_dim: bool,
                 image_patch_max_function_name: str,
                 image_patch_max_smoothing: float,
                 patch_low_concept_n_head: int,
                 patch_low_concept_keep_head_dim: bool,
                 patch_low_concept_max_function_name: str,
                 patch_low_concept_max_smoothing: float):
        super(TopDownHierConceptualPool2d, self).__init__()

        self.high_conceptualizer = Conceptualizer(
            feature_dim=feature_dim,
            concept_dim=concept_dim,
            n_head=image_high_concept_n_head,
            keep_head_dim=image_high_concept_keep_head_dim,
            max_function_name=image_high_concept_max_function_name,
            max_smoothing=image_high_concept_max_smoothing
        )

        self.low_conceptual_pooling = MaskedConceptualPool2d(
            spatial_dim=spatial_dim,
            feature_dim=feature_dim,
            concept_dim=concept_dim,
            image_patch_n_head=image_patch_n_head,
            image_patch_keep_head_dim=image_patch_keep_head_dim,
            image_patch_max_function_name=image_patch_max_function_name,
            image_patch_max_smoothing=image_patch_max_smoothing,
            patch_concept_n_head=patch_low_concept_n_head,
            patch_concept_keep_head_dim=patch_low_concept_keep_head_dim,
            patch_concept_max_function_name=patch_low_concept_max_function_name,
            patch_concept_max_smoothing=patch_low_concept_max_smoothing
        )

    def forward(self, patches: torch.Tensor, low_concepts: torch.Tensor, high_concepts: torch.Tensor, low_high_hierarchy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        overview = patches.mean(dim=1)  # [B, D_q]
        (
            high_conceptual_image,
            image_high_concept_attention_weight
        ) = self.high_conceptualizer(overview, high_concepts)  # [B, D_kv], [B, N_high]

        # [B, N_high]
        image_high_concept_mask = image_high_concept_attention_weight.sign()
        image_low_concept_mask = torch.matmul(
            image_high_concept_mask, low_high_hierarchy.t()
        )  # [B, N_low]

        (
            low_conceptual_image,  # [B, D_kv]
            image_low_concept_attention_weight,  # [B, N_low]
            image_patch_attention_weight,  # [B, H*W]
            patch_low_concept_attention_weight  # [B, H*W, N_low]
        ) = self.low_conceptual_pooling(
            torch.cat([overview.unsqueeze(1), patches], dim=1),
            low_concepts,
            image_low_concept_mask
        )

        patch_high_concept_attention_weight = torch.matmul(
            patch_low_concept_attention_weight, low_high_hierarchy
        )  # [B, H*W, N_high]

        return (
            low_conceptual_image,
            high_conceptual_image,
            image_patch_attention_weight,
            image_low_concept_attention_weight,
            image_high_concept_attention_weight,
            patch_low_concept_attention_weight,
            patch_high_concept_attention_weight
        )


class TopDownHierConceptPoolResNet(nn.Module):
    def __init__(
            self,
            layers,
            num_low_concepts,
            norm_low_concepts,
            num_high_concepts,
            norm_high_concepts,
            concept_dim,
            low_high_max_function,
            output_high_concepts_type,
            detach_low_concepts,
            learnable_hierarchy,
            preset_hierarchy,
            spatial_dim,
            image_high_concept_n_head,
            image_high_concept_keep_head_dim,
            image_high_concept_max_function_name,
            image_high_concept_max_smoothing,
            image_patch_n_head,
            image_patch_keep_head_dim,
            image_patch_max_function_name,
            image_patch_max_smoothing,
            patch_low_concept_n_head,
            patch_low_concept_keep_head_dim,
            patch_low_concept_max_function_name,
            patch_low_concept_max_smoothing,
            image_size,
            width
    ):
        super().__init__()
        self.output_dim = concept_dim
        self.image_size = image_size

        self.image_encoder = ModifiedResNetRemovedAttnPool(
            layers=layers,
            image_size=image_size,
            width=width
        )
        self.feature_dim = width * 32
        self.hierarchical_concepts = HierarchicalConcepts(
            num_low_concepts=num_low_concepts,
            num_high_concepts=num_high_concepts,
            concept_dim=concept_dim,
            low_high_max_function=low_high_max_function,
            output_high_concepts_type=output_high_concepts_type,
            learnable_hierarchy=learnable_hierarchy
        )
        self.norm_low_concepts = norm_low_concepts
        self.norm_high_concepts = norm_high_concepts
        self.detach_low_concepts = detach_low_concepts
        self.preset_hierarchy = preset_hierarchy
        if self.preset_hierarchy:
            assert num_low_concepts % num_high_concepts == 0
            assert output_high_concepts_type == "original_high"
            low_per_high = num_low_concepts // num_high_concepts
            self.low_high_hierarchy = torch.zeros(
                num_low_concepts, num_high_concepts
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            for i in range(num_high_concepts):
                self.low_high_hierarchy[
                    i*low_per_high:(i+1)*low_per_high, i
                ] = 1.0
        self.hierarchical_conceptual_pooling = TopDownHierConceptualPool2d(
            spatial_dim=spatial_dim,
            feature_dim=self.feature_dim,
            concept_dim=concept_dim,
            image_high_concept_n_head=image_high_concept_n_head,
            image_high_concept_keep_head_dim=image_high_concept_keep_head_dim,
            image_high_concept_max_function_name=image_high_concept_max_function_name,
            image_high_concept_max_smoothing=image_high_concept_max_smoothing,
            image_patch_n_head=image_patch_n_head,
            image_patch_keep_head_dim=image_patch_keep_head_dim,
            image_patch_max_function_name=image_patch_max_function_name,
            image_patch_max_smoothing=image_patch_max_smoothing,
            patch_low_concept_n_head=patch_low_concept_n_head,
            patch_low_concept_keep_head_dim=patch_low_concept_keep_head_dim,
            patch_low_concept_max_function_name=patch_low_concept_max_function_name,
            patch_low_concept_max_smoothing=patch_low_concept_max_smoothing
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        (
            low_concepts,
            low_concept_cosine_similarity,
            high_concepts,
            high_concept_cosine_similarity,
            low_high_hierarchy_based_on_similarity
        ) = self.hierarchical_concepts(
            norm_low_concepts=self.norm_low_concepts,
            norm_high_concepts=self.norm_high_concepts,
            detach_low_concepts=self.detach_low_concepts
        )

        if self.preset_hierarchy:
            low_high_hierarchy = self.low_high_hierarchy
        else:
            low_high_hierarchy = low_high_hierarchy_based_on_similarity

        image_patches = self.image_encoder(x).flatten(
            start_dim=2
        ).permute(0, 2, 1)
        assert self.feature_dim == image_patches.size(-1)
        image_patches = image_patches / self.feature_dim ** 0.5

        (
            low_conceptual_image,
            high_conceptual_image,
            image_patch_attention_weight,
            image_low_concept_attention_weight,
            image_high_concept_attention_weight,
            patch_low_concept_attention_weight,
            patch_high_concept_attention_weight
        ) = self.hierarchical_conceptual_pooling(
            patches=image_patches,
            low_concepts=low_concepts,
            high_concepts=high_concepts,
            low_high_hierarchy=low_high_hierarchy
        )

        return {
            "image_features": low_conceptual_image,
            "aux_image_features": high_conceptual_image,
            "image_patch_attention_weight": image_patch_attention_weight,
            "image_low_concept_attention_weight": image_low_concept_attention_weight,
            "image_high_concept_attention_weight": image_high_concept_attention_weight,
            "patch_low_concept_attention_weight": patch_low_concept_attention_weight,
            "patch_high_concept_attention_weight": patch_high_concept_attention_weight,
            "low_concept_cosine_similarity": low_concept_cosine_similarity,
            "high_concept_cosine_similarity": high_concept_cosine_similarity,
            "low_high_hierarchy": low_high_hierarchy
        }


def _build_conceptual_visual_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        conceptual_cfg: CLIPConceptualCfg
) -> nn.Module:

    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    if isinstance(conceptual_cfg, dict):
        conceptual_cfg = CLIPConceptualCfg(**conceptual_cfg)

    if conceptual_cfg.conceptual_type == "TopDownHierConceptualPool2d":
        conceptual_cfg_dict = asdict(conceptual_cfg)
        conceptual_cfg_dict.pop("conceptual_type")
        visual = TopDownHierConceptPoolResNet(
            layers=vision_cfg.layers,
            concept_dim=embed_dim,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
            **conceptual_cfg_dict
        )
    else:
        raise ValueError(
            f"Invalid conceptual type: {conceptual_cfg.conceptual_type}")

    return visual


class PostLayerNormAndProjection(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        return self.proj(self.ln(x))


class ConceptualCLIP(CustomTextCLIP):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_tower: bool,
            vision_cfg: CLIPVisionCfg,
            conceptual_cfg: CLIPConceptualCfg,
            text_tower: bool,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False
    ):
        super().__init__(
            embed_dim=embed_dim,
            vision_tower=vision_tower,
            vision_cfg=vision_cfg,
            text_tower=text_tower,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            cast_dtype=cast_dtype,
            output_dict=output_dict
        )

        if vision_tower:
            self.visual = _build_conceptual_visual_tower(
                embed_dim, vision_cfg, conceptual_cfg
            )
        else:
            self.visual = None

        self.image_post_layernorm_and_projection = PostLayerNormAndProjection(
            embed_dim)
        self.text_post_layernorm_and_projection = PostLayerNormAndProjection(
            embed_dim)

        self.aux_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.aux_image_post_layernorm_and_projection = PostLayerNormAndProjection(
            embed_dim)
        self.aux_text_post_layernorm_and_projection = PostLayerNormAndProjection(
            embed_dim)

    def encode_image(self, image, normalize: bool = False):
        image_outputs = self.visual(image)
        if "image_features" in image_outputs:
            image_features = image_outputs.pop("image_features")
            image_features = self.image_post_layernorm_and_projection(
                image_features)
            image_outputs["image_features"] = F.normalize(
                image_features, dim=-1) if normalize else image_features
        if "aux_image_features" in image_outputs:
            aux_image_features = image_outputs.pop("aux_image_features")
            aux_image_features = self.aux_image_post_layernorm_and_projection(
                aux_image_features)
            image_outputs["aux_image_features"] = F.normalize(
                aux_image_features, dim=-1) if normalize else aux_image_features
        return image_outputs

    def encode_text(self, text, normalize: bool = False, need_aux_text_features: bool = False):
        text_features = self.text(text)
        if self.training and need_aux_text_features:
            aux_text_features = self.aux_text_post_layernorm_and_projection(
                text_features)
        text_features = self.text_post_layernorm_and_projection(
            text_features)
        if self.training and need_aux_text_features:
            return {
                "text_features": F.normalize(text_features, dim=-1) if normalize else text_features,
                "aux_text_features": F.normalize(aux_text_features, dim=-1) if normalize else aux_text_features,
            }
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        assert self.output_dict is True

        image_outputs = self.encode_image(
            image, normalize=True) if image is not None else None
        if image_outputs and "aux_image_features" in image_outputs:
            need_aux_text_features = True
        else:
            need_aux_text_features = False
        text_outputs = self.encode_text(
            text, normalize=True,
            need_aux_text_features=need_aux_text_features) if text is not None else None

        if self.training and need_aux_text_features:
            out_dict = {
                **image_outputs,
                **text_outputs,
                "logit_scale": self.logit_scale.exp(),
                "aux_logit_scale": self.aux_logit_scale.exp(),
            }
        else:
            out_dict = {
                **image_outputs,
                "text_features": text_outputs,
                "logit_scale": self.logit_scale.exp(),
            }
        if self.logit_bias is not None:
            out_dict['logit_bias'] = self.logit_bias
        return out_dict
