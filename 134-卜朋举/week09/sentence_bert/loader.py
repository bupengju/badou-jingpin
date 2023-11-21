import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader


class DataGen(Dataset):

    def __init__(self, data_path, max_length, max_sentence_length):
        super(DataGen, self).__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.max_sentence_length = max_sentence_length
        self.label_map = {"O": 0, "B": 1, "I": 2, }
        self.data = []
        self.load()

    def load(self, ):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for paragraph in f.read().split("\n\n"):
                if paragraph.strip() == "" or "\n" not in paragraph.strip():
                    continue
                i = 0
                for sentence in paragraph.split("\n"):
                    if sentence.strip() == "":
                        continue
                    parts = sentence.split("\t")
                    inputs = parts[0]
                    labels = parts[1][0]
                    self.data.append((parts[0], self.label_map[parts[1][0]]))
                    i += 1
                    if i > self.max_sentence_length:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_data(data_path, vocab_path, batch_size, max_length, max_sentence_length, shuffle=True):
    """
    Load data from data_path and return a DataLoader object.

    :param data_path: str, path to the data file.
    :param vocab_path: str, path to the vocabulary file.
    :param batch_size: int, batch size for DataLoader.
    :param max_length: int, maximum length of each sequence.
    :param shuffle: bool, whether to shuffle the data or not. Default is True.

    :return: DataLoader object.
    """
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    dataset = DataGen(data_path, max_length, max_sentence_length)

    def collate_fn(batch):
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        inputs = tokenizer.batch_encode_plus(
            sentences, padding="max_length", max_length=max_length, truncation=True,  return_tensors="pt")
        input_ids = inputs["input_ids"]
        labels = torch.LongTensor(labels)
        return input_ids, labels
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
