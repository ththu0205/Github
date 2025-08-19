from transformers import AutoTokenizer
from huggingface_hub import HfFolder

# huggingface-cli login
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=HfFolder.get_token()
)

# Prompt ví dụ
prompt = "<|start_header_id|>user<|end_header_id|>\nWhat is the capital of France?\n<|start_header_id|>assistant<|end_header_id|>\n"

# Tokenize
encoded = tokenizer(prompt)

# Lấy input_ids
input_ids = encoded["input_ids"][0]

# Convert input_ids → token
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# In từng token với ID
for token, token_id in zip(tokens, input_ids):
    print(f"{token:<25} -> ID: {token_id.item()}")
