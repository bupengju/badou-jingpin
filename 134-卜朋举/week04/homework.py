# week3作业
# 尝试完成基于词表的全切分（下节课上课会讲）

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
vocabulary = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}

# 待切分文本
test_sentence = "经常有意见分歧"


class FullSeg(object):

    def __init__(self, sentence, vocab):
        self.sentence = sentence
        self.vocab = vocab
        self.length = len(sentence)
        self.path = []
        self.result = []

    def backtrack(self, start_idx):
        if start_idx == self.length:
            self.result.append(self.path[:])
            return

        for end_idx in range(start_idx + 1, self.length + 1):
            word = self.sentence[start_idx:end_idx]
            if word in self.vocab:
                self.path.append(word)
                self.backtrack(end_idx)
                self.path.pop()

    def seg(self):
        self.backtrack(0)
        return self.result


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, vocab):
    full_seg = FullSeg(sentence, vocab)
    return full_seg.seg()


target = all_cut(test_sentence, vocabulary)
for idx, t in enumerate(target):
    print(idx + 1, t)


# 目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]
