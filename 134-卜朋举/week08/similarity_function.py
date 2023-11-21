import numpy as np


def edit_distance(words1, words2):
    result = np.zeros((len(words1) + 1, len(words2) + 1))
    for i in range(len(words1) + 1):
        result[i][0] = i

    for j in range(len(words2) + 1):
        result[0][j] = j

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                d = 0
            else:
                d = 1
            result[i][j] = min(result[i - 1][j] + 1, result[i][j - 1] + 1, result[i - 1][j - 1] + d)
    return result[len(words1)][len(words2)]


def similarity_base_on_ed(words1, words2):
    return 1 - edit_distance(words1, words2) / max(len(words1), len(words2))


def jaccard_distance(words1, words2):
    sim = len(set(words1) & set(words2)) / len(set(words1) | set(words2))
    return 1 - sim


def similarity_base_on_jd(words1, words2):
    return 1 - jaccard_distance(words1, words2)


if __name__ == '__main__':
    a = "北京欢迎你"
    b = "南京欢迎你"
    print(similarity_base_on_ed(a, b))
    print(similarity_base_on_jd(a, b))
