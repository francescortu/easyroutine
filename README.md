# EasyRoutine

This is just a simple collection of routines that I use frequently. I have found that I often need to do the same things over and over again, so I have created this repository to store them. I hope you find them useful.


## Installation


## Interpretability
The interpretability module contains wrapper of huggingface LLM/VLM that help to perform interpretability tasks on the model. Currently, it supports:
- Extract activations of any component of the model
- Perform ablation study on the model during inference
- Perform activation patching on the model during inference

### Simple Tutorial
```python
# First we need to import the HookedModel and the config classes
from easyroutine.interpretability import HookedModel,  ExtractionConfig

hooked_model = HookedModel.from_pretrained(
    model_name="mistral-community/pixtral-12b", # the model name
    device_map = "auto"
)

# Now let's define a simple dataset
dataset = [
    "This is a test",
    "This is another test"
]

tokenizer = hooked_model.get_tokenizer()

dataset = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt") 

cache = hooked_model.extract_cache(
    dataset,
    target_token_positions = ["last"],
    extraction_config = ExtractionConfig(
        extract_resid_out = True
    )
)

````



### Development
For publish the package push a commit with the flag:
  - `[patch]`: x.x.7 -> x.x.8
  - `[minor]`: x.7.x -> x.8.0
  - `[major]`: 2.x.x -> 3.0.0

Example commit: `fix multiple bus [patch]`

-
