# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-7b-1.1")
model = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-7b-1.1")
# from transformers import file_utils
# print(file_utils.default_cache_path)
