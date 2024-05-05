import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_default_device("cuda:0")

model = AutoModelForCausalLM.from_pretrained("D:/ml/code/nlp/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("D:/ml/code/nlp/phi-2", trust_remote_code=True)

sequence = "Instruct:\nWrite a detailed analogy between mathematics and a lighthouse.\nOutput:\n"
inputs = tokenizer(sequence, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)

text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(text)
# Instruct:
# Write a detailed analogy between mathematics and a lighthouse.
# Output:
# Mathematics is like a lighthouse, guiding us through the vast ocean of knowledge.
# Just as a lighthouse emits a steady beam of light to illuminate the way for ships,
# mathematics provides us with a framework to understand and navigate the complexities
# of the world.
# Similar to how a lighthouse helps ships avoid dangerous rocks and navigate safely,
# mathematics equips us with problem-solving skills and logical reasoning to overcome
# challenges and make informed decisions. It serves as a beacon of clarity, shedding
# light on abstract concepts and providing a solid foundation for various disciplines.
# Furthermore, just as a lighthouse stands tall and sturdy, mathematics is built upon
# a solid foundation of axioms and principles. It is a discipline that relies on rigorous
# proofs and logical deductions, ensuring that our conclusions are based on sound reasoning.
# In addition, a lighthouse is constantly evolving and adapting to changing conditions.
# Similarly, mathematics is a dynamic field that
