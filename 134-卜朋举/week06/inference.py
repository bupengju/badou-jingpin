import math
from pathlib import Path

import torch

from nnlm import NNLM
from nnlm import build_vocab


def load_models(input_size, window_size, device, weight_root, vocab_path):
    vocab = build_vocab(vocab_path)

    model_dict = {}
    for f in sorted(Path(weight_root).glob("*.pth")):
        net = NNLM(input_size, vocab)
        net.eval()
        net.input_size = input_size
        net.window_size = window_size
        net.vocab = vocab
        net.load_state_dict(torch.load(f.as_posix()))
        net.to(device)
        model_dict[f.stem] = net

    return model_dict


def calc_ppl(sentence, model, device):
    prob = 0
    with torch.no_grad():
        for end in range(1, len(sentence) - 1):
            start = max(0, end - model.window_size)
            window = sentence[start:end]
            x = [model.vocab.get(word, model.vocab["<UNK>"]) for word in window]
            x = torch.LongTensor([x])
            x = x.to(device)
            target = sentence[end]
            target_idx = model.vocab.get(target, model.vocab["<UNK>"])
            pred = model(x).squeeze()
            prob += math.log10(pred[target_idx])
    return 2 ** (-prob / len(sentence))


def main():
    sentences = ["在全球货币体系出现危机的情况下",
                 "点击进入双色球玩法经典选号图表",
                 "慢时尚服饰最大的优点是独特",
                 "做处女座朋友的人真的很难",
                 "网戒中心要求家长全程陪护",
                 "在欧巡赛扭转了自己此前不利的状态",
                 "选择独立的别墅会比公寓更适合你",
                 ]
    input_size = 64
    win_size = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict = load_models(input_size, win_size, device, "./out", "./data/vocab.txt")
    result_dict = {}
    for sentence in sentences:
        ppl_dict = {}
        for cate, model in models_dict.items():
            ppl_dict[cate] = calc_ppl(sentence, model, device)
        print(ppl_dict)
        pred_cate = sorted(ppl_dict.items(), key=lambda x: x[1], reverse=True)[0]
        result_dict[sentence] = pred_cate[0]
    print(result_dict)


if __name__ == '__main__':
    main()
