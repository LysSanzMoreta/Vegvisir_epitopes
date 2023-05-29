import torch
torch.manual_seed(123)

sentence = 'Life is short, eat dessert first'
dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

d = embedded_sentence.shape[1]
d_q, d_k, d_v = 24, 24, 28
W_query = torch.rand(d_q, d)
W_key = torch.rand(d_k, d)
W_value = torch.rand(d_v, d)

x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
print("query_2")
print(query_2.shape)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)
#unnormalized attention weights ω

keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print(embedded_sentence.T.shape)
print(W_key.shape)
print(keys.shape)

omega_24 = query_2.dot(keys[4]) #nnormalized attention weight for the query and 5th input element (corresponding to index position 4)

print(omega_24)

omega_2 = query_2.matmul(keys.T)

print(omega_2)

attention_weights_2 = torch.nn.functional.softmax(omega_2 / d_k**0.5, dim=0)

print(attention_weights_2)

context_vector_2 = attention_weights_2.matmul(values) #attention-weighted version of our original query input x(2)

print(context_vector_2)