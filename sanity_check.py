import torch
from llama import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sanity_data = torch.load("./sanity_check.data")
# text_batch = ["hello world", "hello neural network for NLP"]
# tokenizer here
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])

# load our model
# run sanity check in float64 on CPU to reduce tiny numerical drift
# across PyTorch versions/platform kernels
llama = load_pretrained("stories42M.pt").cpu().double()
llama.eval()
with torch.no_grad():
    logits, hidden_states = llama(sent_ids)
    ref_logits = sanity_data["logits"].double()
    ref_hidden_states = sanity_data["hidden_states"].double()
    assert torch.allclose(logits, ref_logits, atol=1e-5, rtol=1e-3)
    assert torch.allclose(hidden_states, ref_hidden_states, atol=1e-5, rtol=1e-3)
    print("Your Llama implementation is correct!")
