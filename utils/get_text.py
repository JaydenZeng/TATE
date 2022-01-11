import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import pickle


tokenizer = BertTokenizer.from_pretrained('./bert_en')
model = BertModel.from_pretrained('./bert_en')

def TextEmbedding(tokenizer, model, text):
    # add_special_tokens will add start and end token
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
    return last_hidden_states.squeeze().numpy()

text_embedding = {}

texts = pickle.load(open('./text_dict.pkl', 'rb'))


a= 0


for key in texts:
    tmp = TextEmbedding(tokenizer, model, texts[key])
    text_embedding[key] = tmp
    a += 1
    if a %20 == 0:
        print (a)

pickle.dump(text_embedding, open('./text_emb.pkl', 'wb'))
