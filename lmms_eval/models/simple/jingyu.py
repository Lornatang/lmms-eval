# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("jingyu")
class Jingyu(lmms):
    """Jingyu multimodal model via standard Transformers Auto* APIs.

    Loads through checkpoint ``auto_map``:
      - ``AutoModelForCausalLM`` -> ``JingyuForConditionalGeneration``
      - ``AutoProcessor`` -> Jingyu processor from ``processing_jingyu``
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the Jingyu model wrapper and load weights via Transformers Auto APIs.

        Args:
            pretrained (str): Hugging Face model id or local checkpoint path.
            device (str, optional): Device string used when not under multi-process accelerate.
            device_map (str, optional): Device map for ``from_pretrained`` (e.g. ``auto`` or a CUDA device).
            batch_size (int or str, optional): Per-GPU batch size for generation.
            use_cache (bool): Whether to enable KV cache during generation.
            attn_implementation (str, optional): Attention backend; one of ``None``, ``flash_attention_2``, ``sdpa``, or ``eager``.
            dtype (str): Torch dtype name for model weights (default ``bfloat16``).
            trust_remote_code (bool): Whether to trust remote code from the checkpoint ``auto_map``.
            system_prompt (str, optional): Optional system prompt prepended to each conversation.
            interleave_visuals (bool, optional): If True, place images at ``<image N>`` placeholders in text.
            reasoning_prompt (str, optional): Optional suffix appended to the user text prompt.
            **kwargs: Must be empty; unexpected keys raise ``AssertionError``.

        Raises:
            ValueError: If ``attn_implementation`` is not in the allowed set.
            AssertionError: If unexpected ``kwargs`` are provided.
        """
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": utils.get_dtype(dtype),
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"
            self._tokenizer = self.processor.tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
            self._tokenizer.padding_side = "left"

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.dtype = dtype

        self._config = self.model.config
        self._max_length = getattr(self._config, "tokenizer_model_max_length", None) or getattr(self._config, "max_position_embeddings", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _build_generate_kwargs(self, gen_kwargs):
        """Build ``model.generate()`` kwargs by merging user values with defaults.

        Args:
            gen_kwargs (dict): Per-request generation keyword arguments from the task.

        Returns:
            dict: Keyword arguments ready to pass to ``model.generate()``.
        """
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val

        return generate_kwargs

    def _build_messages(self, context, visuals):
        """Build a Jingyu multimodal conversation for one sample.

        Args:
            context (str): User text prompt for the sample.
            visuals (list or None): Visual inputs for the sample (images and/or video paths), or None.

        Returns:
            list: Chat messages suitable for ``processor.apply_chat_template``.
        """
        if "<image>" in context:
            context = context.replace("<image>", "")

        if self.reasoning_prompt:
            context = context.strip() + self.reasoning_prompt

        processed_visuals = []
        if visuals is not None:
            for visual in visuals:
                if isinstance(visual, Image.Image):
                    processed_visuals.append({"type": "image", "image": visual})
                elif isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    processed_visuals.append({"type": "video", "video": visual})
                elif isinstance(visual, str):
                    processed_visuals.append({"type": "image", "image": visual})

        message = []
        if self.system_prompt:
            message.append({"role": "system", "content": self.system_prompt})

        if self.interleave_visuals is False:
            content = processed_visuals + [{"type": "text", "text": context}]
        else:
            image_placeholders = re.findall(r"<image \d+>", context)
            content = []
            text_parts = re.split(r"<image \d+>", context)
            if text_parts[0]:
                content.append({"type": "text", "text": text_parts[0]})

            for placeholder_idx, placeholder in enumerate(image_placeholders):
                img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                if processed_visuals and image_idx < len(processed_visuals):
                    content.append(processed_visuals[image_idx])
                if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                    content.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

            if not image_placeholders:
                content = processed_visuals + [{"type": "text", "text": context}]

        message.append({"role": "user", "content": content})
        return message

    def _preprocess_chunk(self, chunk):
        """Preprocess one batch chunk on CPU via Jingyu ``processor.apply_chat_template``.

        Args:
            chunk (tuple): A batched list of request argument tuples from the collator.

        Returns:
            tuple: ``(inputs, contexts, gen_kwargs, until)`` with tensors still on CPU.

        Raises:
            ValueError: If ``gen_kwargs['until']`` is not a ``str`` or ``list``.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        # Drop newline stop strings: many chat models emit a leading "\n" before the answer,
        # and splitting on "\n" would incorrectly yield an empty string.
        until = [item for item in until if item not in ("\n", "\n\n")]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        batched_messages = []
        for i, context in enumerate(contexts):
            if self.reasoning_prompt:
                contexts[i] = context.strip() + self.reasoning_prompt
            batched_messages.append(self._build_messages(context, visual_list[i]))

        processor_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.batch_size > 1:
            processor_kwargs["padding"] = True

        inputs = self.processor.apply_chat_template(batched_messages, **processor_kwargs)
        return inputs, contexts, gen_kwargs, until

    @property
    def batch_size(self):
        """Returns the per-GPU batch size used for generation."""
        return self.batch_size_per_gpu

    @property
    def config(self):
        """Returns the Hugging Face model config associated with the loaded checkpoint."""
        return self._config

    @property
    def device(self):
        """Returns the primary torch device used by this process."""
        return self._device

    @property
    def eot_token_id(self):
        """Returns the end-of-text token id used as the default stop token."""
        return self.tokenizer.eos_token_id

    def flatten(self, input):
        """Flatten a nested list of visual inputs into a single-level list.

        Args:
            input (list): Nested iterable of visual items (e.g. list of lists of images).

        Returns:
            list: A flat list containing all leaf visual items.
        """
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate model responses for a list of evaluation requests.

        Overlaps CPU preprocessing of the next batch with GPU generation of the current batch via a single-worker ``ThreadPoolExecutor``.

        Args:
            requests (List[Instance]): Evaluation instances whose ``args`` provide context, visuals, and generation kwargs.

        Returns:
            List[str]: Generated answer strings in the original request order.
        """
        res = []

        def _collate(x):
            """Collate key that sorts by descending tokenized context length.

            Args:
                x (tuple): A request argument tuple whose first element is the context string.

            Returns:
                tuple: A pair ``(-token_length, context)`` used by the collator.
            """
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._preprocess_chunk, chunks[0]) if chunks else None

            for idx in range(len(chunks)):
                inputs, contexts, gen_kwargs, until = future.result()

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1])

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # pixel_values often need model dtype; keep integer tensors unchanged
                model_dtype = next(self.model.parameters()).dtype
                for key, value in list(inputs.items()):
                    if torch.is_floating_point(value):
                        inputs[key] = value.to(dtype=model_dtype)

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                cont = self.model.generate(**inputs, **generate_kwargs)

                # Some Jingyu checkpoints return full sequences (prompt + new tokens); others return
                # only newly generated tokens. Trim only when the prompt prefix is present.
                generated_ids_trimmed = []
                for in_ids, out_ids in zip(inputs["input_ids"], cont):
                    in_len = in_ids.shape[-1]
                    if out_ids.shape[-1] > in_len and torch.equal(out_ids[:in_len], in_ids.to(out_ids.device)):
                        generated_ids_trimmed.append(out_ids[in_len:])
                    elif out_ids.shape[-1] > in_len:
                        generated_ids_trimmed.append(out_ids[in_len:])
                    else:
                        generated_ids_trimmed.append(out_ids)
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    ans = ans.strip()
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans.strip()

                for ans, context in zip(answers, contexts):
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate multi-round responses for evaluation requests.

        Args:
            requests (list): Evaluation instances for multi-round dialogue generation.

        Returns:
            List[str]: Generated answer strings for each request.

        Raises:
            NotImplementedError: Always; multi-round generation is not implemented yet.
        """
        raise NotImplementedError("TODO: Implement multi-round generation for Jingyu")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood scores for continuations given contexts.

        Args:
            requests (List[Instance]): Evaluation instances with context/continuation pairs.

        Returns:
            List[Tuple[float, bool]]: A list of ``(logprob, is_greedy)`` pairs for each request.

        Raises:
            NotImplementedError: Always; log-likelihood is not implemented for Jingyu.
        """
        raise NotImplementedError("Loglikelihood is not implemented for Jingyu models")

    @property
    def max_length(self):
        """Returns the maximum sequence length supported by the model/tokenizer."""
        return self._max_length

    @property
    def model(self):
        """Returns the underlying model, unwrapping Accelerate wrappers when present."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def rank(self):
        """Returns the local process rank in distributed evaluation."""
        return self._rank

    @property
    def tokenizer(self):
        """Returns the tokenizer associated with the loaded processor/checkpoint."""
        return self._tokenizer

    @property
    def world_size(self):
        """Returns the number of processes used for distributed evaluation."""
        return self._world_size
