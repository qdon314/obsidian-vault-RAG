from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag.adapters.chunking.proposition._parsing import _extract_json_list, _parse_json_list_loose


class Propositionizer(Protocol):
    def propositionize_many(self, inputs: list[str]) -> list[list[str]]:
        """Convert a list of passage strings into lists of proposition strings."""
        ...
        
    def get_config(self) -> dict[str, Any]:
        """Backend-specific config for manifests/debugging."""
        ...

@dataclass
class T5Propositionizer:
    """
    Wrapper around a seq2seq model that converts prose passages into
    lists of atomic, self-contained propositions.

    This class is performance-sensitive:
    - flan-t5-large is ~770M params and autoregressive on the decoder
    - generation speed depends heavily on:
        * device (MPS > CPU on Apple Silicon)
        * batch size
        * max_new_tokens
        * input sequence length

    The defaults here are tuned for Apple Silicon (MPS) indexing workloads.
    """

    # HuggingFace model identifier
    model_name: str = "chentong00/propositionizer-wiki-flan-t5-large"

    # Device to run inference on:
    #   - "mps"  (Apple Silicon GPU, preferred on Mac)
    #   - "cuda" (NVIDIA GPU)
    #   - "cpu"  (fallback, slow)
    # If None, auto-detected in __post_init__.
    device: str | None = None

    # Number of passages processed in one forward/generation call.
    # On MPS, 2-4 is usually optimal. On CUDA, you can often go higher.
    batch_size: int = 4

    # Maximum number of *generated* tokens per passage.
    # Propositions are short; keeping this small dramatically improves speed.
    max_new_tokens: int = 128

    # Maximum number of *input* tokens per passage.
    # This caps encoder cost and prevents pathological tokenization
    # (tables, lists, weird markdown).
    max_input_tokens: int = 512

    def __post_init__(self) -> None:
        """
        Load tokenizer and model, select device, and put model in inference mode.

        IMPORTANT:
        - This should happen ONCE per process (e.g. in your container),
          not once per document.
        """
        # ----------------------------
        # Device selection
        # ----------------------------
        # Prefer MPS on Apple Silicon, then CUDA, then CPU.
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        # ----------------------------
        # Tokenizer
        # ----------------------------
        # Tokenizer is lightweight and shared across batches.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # ----------------------------
        # Model
        # ----------------------------
        # Load the seq2seq model onto the chosen device.
        # NOTE: We do not force fp16 on MPS; PyTorch handles this internally.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        # Disable dropout, layernorm stats, etc.
        self.model.eval()

        # Useful sanity check during debugging
        print(f"[Propositionizer] device={self.device}, "
              f"batch_size={self.batch_size}, "
              f"max_new_tokens={self.max_new_tokens}, "
              f"max_input_tokens={self.max_input_tokens}")

    def propositionize_many(self, inputs: list[str]) -> list[list[str]]:
        """
        Convert a list of passage strings into lists of proposition strings.

        Each input passage should already be formatted as:
            "Title: <title>. Section: <heading>. Content: <passage text>"

        This method:
        1) Batches inputs for throughput
        2) Tokenizes with truncation (input length cap)
        3) Runs deterministic generation (no sampling, no beam search)
        4) Parses the model output into JSON lists of propositions

        Returns:
            One list of propositions per input passage.
            Empty lists are possible if parsing fails.
        """
        out: list[list[str]] = []

        # Process inputs in fixed-size batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]

            # ----------------------------
            # Tokenization
            # ----------------------------
            # - padding=True allows batching
            # - truncation=True + max_length caps encoder cost
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )

            # Move tensors to the target device (mps/cuda/cpu)
            toks = {k: v.to(self.device) for k, v in toks.items()}

            # ----------------------------
            # Generation
            # ----------------------------
            # inference_mode() is slightly faster than no_grad()
            # and disables version counter updates.
            with torch.inference_mode():
                gen = self.model.generate(
                    **toks,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,   # deterministic, faster
                    num_beams=1,       # beam search is *very* slow here
                )

            # ----------------------------
            # Decode + parse
            # ----------------------------
            # Convert token IDs back to strings and parse JSON arrays.
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            out.extend([_extract_json_list(t) for t in decoded])

        return out
    
    def get_config(self) -> dict[str, Any]:
        """Return backend-specific config for manifests/debugging."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
        }
    
@dataclass
class GGUFPropositionizer:
    """
    !!! Experimental !!!
    Propositionizer powered by a GGUF decoder-only model via llama.cpp.

    This is *not* the same model as the T5 propositionizer.
    Instead we prompt an instruction-tuned LLM to output a JSON list of atomic propositions.

    Works well for:
    - offline indexing
    - CPU/Apple Silicon Metal acceleration
    - controllable output formatting

    You must provide a GGUF model file path.
    """
    model_path: str

    # llama.cpp runtime knobs
    n_ctx: int = 256
    n_threads: int = 4
    n_gpu_layers: int = 0  # on Apple Silicon, Metal will offload; on CPU-only set 0
    verbose: bool = False

    # generation knobs
    temperature: float = 0.0
    max_tokens: int = 256  # output length (not input)
    top_p: float = 1.0

    # prompt knobs
    system_prompt: str = (
        "You convert passages into atomic, self-contained factual propositions.\n"
        "Return ONLY a valid JSON array of strings.\n"
        "No markdown, no commentary.\n"
    )

    def __post_init__(self) -> None:
        # Late import so you can keep this file in your repo without requiring llama-cpp-python immediately
        from llama_cpp import Llama  # type: ignore

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
            n_batch_size=1,  # llama.cpp internal batch size; keep 1 for low memory usage
        )

    def propositionize_many(self, inputs: list[str]) -> list[list[str]]:
        out: list[list[str]] = []
        for inp in inputs:
            out.append(self._one(inp))
        return out

    def _one(self, inp: str) -> list[str]:
        """
        Single-input inference. llama.cpp doesn't do true GPU batching like HF,
        but this is usually still fast for GGUF on Apple Silicon.
        """
        prompt = self._format_prompt(inp)

        # Use chat-completions style API if available, otherwise fallback to completion.
        # llama-cpp-python supports create_chat_completion for many models.
        try:
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            text = resp["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: raw completion
            resp = self.llm(
                f"{self.system_prompt}\n\n{prompt}\n",
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=["\n\n\n"],  # crude stop
            )
            text = resp["choices"][0]["text"]

        props = _parse_json_list_loose(text)
        return props
    
    def get_config(self) -> dict[str, Any]:
        """Return backend-specific config for manifests/debugging."""
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def _format_prompt(self, inp: str) -> str:
        """
        inp already looks like:
        Title: ...
        Section: ...
        Content: ...

        We add explicit output constraints and examples.
        """
        return (
            "Extract atomic propositions from the passage below.\n"
            "Rules:\n"
            "1) Each proposition must be standalone and unambiguous.\n"
            "2) Preserve named entities and numbers.\n"
            "3) Split compound sentences.\n"
            "4) Output ONLY JSON: a list of strings.\n\n"
            f"PASSAGE:\n{inp}\n\n"
            "OUTPUT JSON ARRAY:"
    )