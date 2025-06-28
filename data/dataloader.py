import tiktoken
from torch.utils.data import DataLoader

from data.dataset import GPTDataset

# create_dataloader is a utility function to create a DataLoader for the GPTDataset.
# It takes a text input, batch size, maximum sequence length, stride, shuffle option,
# whether to drop the last batch, and number of workers for data loading.
# It is used in the training loop to load the data in batches for training the GPT model.
def create_dataloader(txt, batch_size=4, max_length=4, stride=128, shuffle=True, drop_last=True, num_workers=4):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader