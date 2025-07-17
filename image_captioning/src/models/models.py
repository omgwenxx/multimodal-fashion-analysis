from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from peft import PeftModel
import PIL
from typing import Union, List
from peft import LoraConfig, get_peft_model

llava_linear_layers = [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "o_proj",
    "v_proj",
    "k_proj",
]
llava_qv_layers = ["q_proj", "v_proj"]


class BLIP2Model:
    """
    Model class wrapper for blip2 model with optional PEFT support.
    supports following models:
    * Salesforce/blip2-opt-2.7b
    * Salesforce/blip2-opt-6.7b
    * Salesforce/blip2-flan-t5-xl
    * Salesforce/blip2-flan-t5-xxl

    Tokenizer: GPT2TokenizerFast
    Processor: Blip2Processor
    """

    def __init__(
        self,
        base_checkpoint: str,
        adapter_checkpoint: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initializes the BLIP2 model with the specified parameters.
        Parameters:
            base_checkpoint (str): The base model checkpoint to load.
            adapter_checkpoint (str, optional): The PEFT adapter checkpoint to load. Defaults to None
            torch_dtype (torch.dtype, optional): The torch dtype to use. Defaults to torch.bfloat16.
            device (str, optional): The device to load the model on. Defaults to "cuda
        """
        self.torch_dtype = torch_dtype
        self.device = device

        print(f"Loading BLIP2 model {base_checkpoint.split('/')[-1]} to {device} with dtype {torch_dtype}")

        # Load the base model
        base_model = Blip2ForConditionalGeneration.from_pretrained(base_checkpoint, torch_dtype=self.torch_dtype).to(self.device)

        # If adapter checkpoint is provided, load PEFT model; otherwise, use base model directly
        if adapter_checkpoint:
            print(f"Applying PEFT adapter from {adapter_checkpoint.split('/')[-1]}")
            self.model = PeftModel.from_pretrained(base_model, model_id=adapter_checkpoint, device=self.device)
        else:
            self.model = base_model

        self.processor = Blip2Processor.from_pretrained(base_checkpoint)
        self.tokenizer = self.processor.tokenizer

    def generate_text(
        self,
        images: Union[PIL.Image, torch.Tensor, List[Union[PIL.Image, torch.Tensor]]],
        max_length: int = 100,
    ):
        """
        Generate text based on the given image(s). Supports single or multiple images.

        Parameters:
            images (Union[PIL.Image, torch.Tensor, List[Union[PIL.Image, torch.Tensor]]]):
                The input image or list of images for generating text. Images can be a single
                PIL image, a single torch tensor, or a list containing PIL images and/or tensors.
            max_length (int, optional): The maximum length of the generated text. Defaults to 100.

        Returns:
            Union[str, List[str]]: The generated text for a single image or a list of texts for multiple images.

        """
        inputs = self.processor(images, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(**inputs, max_length=max_length)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text


class LlavaModel:
    """
    Model class wrapper for llava model
    supports following models:
    * llava-hf/llava-1.5-7b-hf
    * llava-hf/llava-1.5-13b-hf (if it fits)
    """

    def __init__(
        self,
        base_checkpoint: str,
        adapter_checkpoint: str = None,
        prompt: str = "Describe the appearance of the clothing item",
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
    ):  
        """
        Initializes the Llava model with the specified parameters.
        Parameters:
            base_checkpoint (str): The base model checkpoint to load.
            adapter_checkpoint (str, optional): The PEFT adapter checkpoint to load. Defaults to None
            prompt (str, optional): The prompt to use for text generation. Defaults to "Describe the appearance of the clothing item".
            torch_dtype (torch.dtype, optional): The torch dtype to use. Defaults to torch.bfloat16.
            device (str, optional): The device to load the model on. Defaults to "cpu".
        """
        self.torch_dtype = torch_dtype
        self.device = device

        print(f"Loading Llava model {base_checkpoint.split('/')[-1]} to {device} with dtype {torch_dtype}")

        # Load the base model
        base_model = LlavaForConditionalGeneration.from_pretrained(base_checkpoint, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True).to(device)

        # If adapter checkpoint is provided, load PEFT model; otherwise, use base model directly
        if adapter_checkpoint:
            print(f"Applying PEFT adapter from {adapter_checkpoint.split('/')[-1]}")
            self.model = PeftModel.from_pretrained(base_model, model_id=adapter_checkpoint, device=self.device)
        else:
            self.model = base_model
        self.processor = AutoProcessor.from_pretrained(base_checkpoint)
        self.tokenizer = self.processor.tokenizer
        self.prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    def generate_text(self, image: Union[PIL.Image, torch.Tensor], max_new_tokens: int = 256):
        """
        Generate text based on the given image.
        Parameters:
            image (PIL.Image or torch.Tensor): The input image for generating text.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 256.
        Returns:
            str: The generated text.
        """
        inputs = self.processor(self.prompt, image, return_tensors="pt").to(self.device, self.torch_dtype)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[0][2:], skip_special_tokens=True)

    def find_all_linear_names(self):
        """
        Finds all linear layer names in the model that are suitable for LoRA.
        Returns:
            List[str]: A list of names of linear layers that can be used with LoRA.
        """
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["multi_modal_projector", "vision_model"]
        for name, module in self.model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")

        return list(lora_module_names)


def get_lora_model(
    model: PreTrainedModel,
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_layers: str | List[str] = "all-linear",
):
    """
    Get a PEFT model with LoRA support.
    Parameters:
        model (PreTrainedModel): The model.
        r (int, optional): The r value for LoRA. Defaults to 32.
        lora_alpha (int, optional): The alpha value for LoRA. Defaults to 64.
        lora_dropout (float, optional): The dropout rate for LoRA. Defaults to 0.05.
        lora_layers (str|List[str], optional): The target layers for LoRA. Defaults to "all-linear".
    Returns:
        PeftModel: The PEFT model with LoRA support.
    """
    config = LoraConfig(
        use_rslora=True,
        r=r,  # default 8
        lora_alpha=lora_alpha,  # default 8
        lora_dropout=lora_dropout,  # default 0
        target_modules=lora_layers,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    print("Training layers", model.active_peft_config.target_modules)
    return model


def get_model(
    checkpoint: str,
    adapter_checkpoint: str = None,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """
    Factory method to get the model based on the checkpoint.
    Parameters:
        checkpoint (str): The model checkpoint to load.
        dtype (torch.dtype, optional): The torch dtype to use. Defaults to torch.bfloat16.
        device (str, optional): The device to use. Defaults to "cuda".
    """
    if "blip" in checkpoint:
        model = BLIP2Model(
            checkpoint,
            adapter_checkpoint=adapter_checkpoint,
            torch_dtype=dtype,
            device=device,
        )
    elif "llava" in checkpoint:
        model = LlavaModel(checkpoint, torch_dtype=dtype, device=device)
    else:
        print(checkpoint)
        raise ValueError("Invalid checkpoint type")
    return model
