import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig


pretrained_model_name_or_path = "Phi-3-mini-128k-instruct"


model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    num_beams=1,
    temperature=0.8,
    top_k=40,
    top_p=0.8,
)

generation_args = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "num_beams": 1,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.8,
    "return_full_text": False,
}

messages = [
    {
        "role": "system",
        "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.",
    },
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    },
    {
        "role": "assistant",
        "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
    },
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

output = pipe(messages, **generation_args)
print(output[0]["generated_text"])
