from transformers import BertTokenizer
from config import MODEL_NAME

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
