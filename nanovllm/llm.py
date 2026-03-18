from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


class LLM(LLMEngine):
    """User-facing LLM class that wraps LLMEngine.

    This class acts as the stable public API for text generation.  It decouples
    the user-facing interface from the internal engine implementation, providing
    input validation and clear error messages that insulate callers from internal
    changes to :class:`~nanovllm.engine.llm_engine.LLMEngine`.

    Args:
        model (str): Path to the model directory or a HuggingFace model ID.
        max_num_batched_tokens (int): Maximum total tokens processed in one step.
        max_num_seqs (int): Maximum number of sequences processed concurrently.
        max_model_len (int): Maximum sequence length supported by the model.
        gpu_memory_utilization (float): Fraction of GPU memory to allocate for
            the KV cache, between 0 (exclusive) and 1 (inclusive).
        tensor_parallel_size (int): Number of GPUs to use for tensor parallelism.
        enforce_eager (bool): When ``True``, disables CUDA graphs and uses eager
            PyTorch execution (useful for debugging).

    Example::

        from nanovllm import LLM, SamplingParams

        llm = LLM("/path/to/model", tensor_parallel_size=1)
        outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=128))
        print(outputs[0]["text"])
    """

    def __init__(self, model: str, **kwargs):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("'model' must be a non-empty string (path or model ID).")

        tp = kwargs.get("tensor_parallel_size", 1)
        if not isinstance(tp, int) or tp < 1:
            raise ValueError("'tensor_parallel_size' must be a positive integer.")

        gpu_util = kwargs.get("gpu_memory_utilization", 0.9)
        if not isinstance(gpu_util, float) or not (0.0 < gpu_util <= 1.0):
            raise ValueError(
                "'gpu_memory_utilization' must be a float in the range (0, 1]."
            )

        super().__init__(model, **kwargs)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Generate text completions for a list of prompts.

        Args:
            prompts: A list of string prompts or pre-tokenized token-ID lists.
            sampling_params: A single :class:`~nanovllm.sampling_params.SamplingParams`
                shared across all prompts, or one instance per prompt.
            use_tqdm: Whether to display a progress bar during generation.

        Returns:
            A list of dicts (one per prompt, in input order), each containing:

            * ``"text"`` – the decoded output string.
            * ``"token_ids"`` – the raw output token IDs.

        Raises:
            ValueError: If ``prompts`` is empty, or if ``sampling_params`` is a
                list whose length does not match the number of prompts.
        """
        if not prompts:
            raise ValueError("'prompts' must be a non-empty list.")

        if isinstance(sampling_params, list) and len(sampling_params) != len(prompts):
            raise ValueError(
                f"When 'sampling_params' is a list, its length "
                f"({len(sampling_params)}) must equal the number of prompts "
                f"({len(prompts)})."
            )

        return super().generate(prompts, sampling_params, use_tqdm)

