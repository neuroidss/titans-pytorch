# FILE: titans_pytorch/__init__.py

from titans_pytorch.neural_memory import (
    NeuralMemory,
    NeuralMemState,
    mem_state_detach
)

from titans_pytorch.memory_models import (
    MemoryMLP,
    MemoryAttention,
    FactorizedMemoryMLP,
    MemorySwiGluMLP,
    GatedResidualMemoryMLP
)

from titans_pytorch.mac_transformer import (
    MemoryAsContextTransformer,
    SegmentedAttention,
    FeedForward,
    QwenMimicMLP
)

from titans_pytorch.mag_transformer import (
    MAGTransformer,
    MAGCache,
    MAGLayerCache,
    MAGLayer
)

from titans_pytorch.mal_transformer import (
    MALTransformer,
    MALCache,
    MALLayerCache,
    MALLayer
)

from titans_pytorch.tta_transformer import (
    TTATransformer,
    TTACache,
    TTALayerCache,
    TTALayer
)

from titans_pytorch.axial_positional_embedding import (
    ContinuousAxialPositionalEmbedding
)

# Added for Qwen Direct Copy
from titans_pytorch.qwen_direct_copy_transformer import (
    Qwen3CopiedForCausalLM,
    Qwen3CopiedModel,
    Qwen3CopiedRMSNorm,
    Qwen3CopiedMLP,
    Qwen3CopiedAttention,
    Qwen3CopiedDecoderLayer,
    Qwen3CopiedRotaryEmbedding,
    Qwen3Config # Make sure Qwen3Config is correctly sourced or defined
)
