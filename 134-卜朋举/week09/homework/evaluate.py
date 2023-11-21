import torch


def test(model, test_data):
    model.eval()

    with torch.no_grad():
        watch_acc = []
        for x, y in test_data:
            logits = model(x)
            pred = logits.argmax(dim=-1)
            selected = x["attention_mask"] == 1
            y = y[selected]
            correct = (pred == y).sum().item()
            total = len(y)
            watch_acc.append(correct / total)
        watch_acc = sum(watch_acc) / len(watch_acc)
    return watch_acc
