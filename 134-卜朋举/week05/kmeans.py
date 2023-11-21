import numpy as np


class KMeans(object):

    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def init(self, x):
        assert self.n_cluster > 0 or self.n_cluster <= x.shape[0], "n_cluster设置错误!"
        idxes = np.random.choice(np.arange(x.shape[0]), self.n_cluster)
        central = []
        for i in idxes:
            central.append(x[i].tolist())
        return central

    def helper(self, x, central):
        result = [[] for _ in range(self.n_cluster)]
        for item in x:
            distance_min = float("inf")
            index = -1
            for i in range(len(central)):
                distance = self.calc_distance(item, central[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index].append(item.tolist())

        new_central = []
        for item in result:
            new_central.append(np.mean(item, axis=0).tolist())

        if np.all(central == new_central):
            return result, central, self.calc_cluster_total_dist(new_central, result)

        return self.helper(x, np.array(new_central))

    def fit(self, x):
        central = self.init(x)
        return self.helper(x, central)

    def calc_cluster_total_dist(self, central, result):
        total_dist = 0
        for i in range(len(central)):
            for j in range(len(result[i])):
                total_dist += self.calc_distance(result[i][j], central[i])

        return total_dist

    def calc_distance(self, x1, x2):
        dist = 0
        for i in range(len(x1)):
            dist += np.square(x1[i] - x2[i])
        return np.sqrt(dist)


if __name__ == '__main__':
    data = np.random.rand(100, 8)
    kmeans = KMeans(8)
    result, centers, distances = kmeans.fit(data)
    print(result)
    print(centers)
    print(distances)
