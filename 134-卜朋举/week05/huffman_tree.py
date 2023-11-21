import heapq  # 小堆


# 定义一个节点类
class Node(object):

    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # 重写比较操作符，以便在heapq中使用(升序排序)
    def __lt__(self, other):
        return self.freq < other.freq


# 定义霍夫曼树类
class HuffmanTree(object):

    def __init__(self, freq_map):
        self.freq_map = freq_map
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # 建立霍夫曼树
    def build_tree(self):
        for char, freq in self.freq_map.items():
            node = Node(char, freq)
            heapq.heappush(self.heap, node)

        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    # 用于生成霍夫曼编码
    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)


def main():
    # 1.构建字典
    words = "你 我 他 你们 我们 他们 它们"
    frequency = "50 10 8 7 6 3 2"
    word_to_id = {word: i for i, word in enumerate(words.split(" "))}
    words_freq = {word_to_id[word]: int(freq) for word, freq in zip(words.split(" "), frequency.split(" "))}
    print(word_to_id)
    print(words_freq)

    huffman_tree = HuffmanTree(words_freq)
    huffman_tree.build_tree()
    huffman_tree.make_codes()
    print(huffman_tree.codes)
    print(huffman_tree.reverse_mapping)


if __name__ == '__main__':
    main()
