from transformers import BertTokenizer

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("./data/bert-base-chinese")
    sentence = "幸亏我一把把把住了"
    tokens = tokenizer.tokenize(sentence)
    print("分字：", tokens)

    encoding = tokenizer.encode(sentence)
    print("编码：", encoding)

    sentence1 = "爸爸的爸爸叫爷爷"
    sentence2 = "妈妈的妈妈叫姥姥"
    encoding = tokenizer.encode(sentence1, sentence2)
    print("文本对编码：", encoding)
    encoding = tokenizer.encode_plus(sentence1, sentence2)
    print("全部编码：", encoding)
