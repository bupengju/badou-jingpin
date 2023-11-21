import torch


def predict(sentence, vocab_path, model_path, max_length, max_sentence_length):
    """
    Predict the label of the sentence.

    :param sentence: str, the sentence to be predicted.
    :param vocab_path: str, path to the vocabulary file.
    :param model_path: str, path to the model file.
    :param max_length: int, maximum length of each sequence.
    :param max_sentence_length: int, maximum length of each sentence.
    :return: list of str, the labels of the sentence.
    """
    tokenizer = BertTokenizer.from_pretrained(config["pretrained_path"])
    model = SentenceBert(config["class_num"], config["pretrained_path"],
                         config["hidden_size"], config["dropout_rate"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.batch_encode_plus([sentence], padding="max_length", max_length=max_length, return_tensors="pt")
        input_ids = inputs["input_ids"]
        logits = model(input_ids)
        predict = torch.argmax(logits, dim=-1)
        return predict
