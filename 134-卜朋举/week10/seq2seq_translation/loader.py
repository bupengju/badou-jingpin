from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from preprocessing import normalize_string


class LangDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
    
    def load_data(self, ):
        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                eng_sentence, fra_sentence = line.strip().split("\t")[:2]
                eng_sentence, fra_sentence = normalize_string(eng_sentence), normalize_string(fra_sentence)
                data.append((eng_sentence, fra_sentence))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    eng_sentences, fra_sentences = zip(*batch)
    return eng_sentences, fra_sentences
