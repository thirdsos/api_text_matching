import numpy as np
def edit_distance(corpus, word):
    distances = np.zeros((len(corpus)), dtype=np.uint8)
    for i, c in enumerate(corpus):
        distances[i] = _edit_distance(c, word)
    return distances
    
def _edit_distance(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    DP = [[0 for i in range(len1 + 1)] for j in range(2)]
    for i in range(0, len1 + 1):
        DP[0][i] = i
    for i in range(1, len2 + 1):
        for j in range(0, len1 + 1):
            if (j == 0):
                DP[i % 2][j] = i
            elif(str1[j - 1] == str2[i-1]):
                DP[i % 2][j] = DP[(i - 1) % 2][j - 1]
            else:
                DP[i % 2][j] = (1 + min(DP[(i - 1) % 2][j], min(DP[i % 2][j - 1], DP[(i - 1) % 2][j - 1])))
    return DP[len2 % 2][len1]