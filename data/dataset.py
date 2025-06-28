import torch
from torch.utils.data import Dataset

# GPTDataset is a custom dataset class for the GPT model
# It takes a text input, tokenizes it using the GPT tokenizer,
# and creates input-target pairs for training.
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1:i + max_length + 1]
            # `:` is a slice operator, here it takes a slice of the list and returns a new sublist from 
            # i to i + max_length
            # i + max_length is exclusive, so it will not include the token at that index
            # e.g. if i = 0 and max_length = 4, input_chunk will be token_ids[0:4] which is the first 4 tokens
            # if i = 1 and max_length = 4, input_chunk will be token_ids[1:5] which is the second to fifth tokens
            # this is how we create the input chunk

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            # when we append the input and target chunks, we convert them to tensors of type long
            # adding a new tensor as an entry in a list of training samples

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.inp