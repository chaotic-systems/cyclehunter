import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import eig


def eigenV(s):
    ret = []
    a = np.array([[(-3 * 12.25 * (1.46385) ** 2 + 14.25) ** 2 - 1, -(-3 * 12.25 * (1.46385) ** 2 + 14.25)],
                  [(-3 * 12.25 * (1.46385) ** 2 + 14.25), -1]])
    w, v = eig(a)
    if abs(w[0]) < abs(w[1]):
        ret.append(v[1])
        ret.append(v[0])
    else:
        ret.append(v[0])  # unstable
        ret.append(v[1])  # stable
    return (ret)


def soldiers(x, y, n, m):
    ret = []
    for i in range(-n, n):
        ret.append((x * 3.0 / n * i + m, y * 3.0 / n * i - m))
    return (ret)


def foreMap(s, M):
    ret = []
    for i in M:
        ret.append((s * i[0] - i[1] - (s - 2) * i[0] * i[0] * i[0], i[0]))
    return (ret)


def backMap(s, M):
    ret = []
    for i in M:
        ret.append((i[1], s * i[1] - i[0] - (s - 2) * i[1] * i[1] * i[1]))
    return (ret)


a = eigenV(5.5)

uV1 = soldiers(a[0][0], a[0][1], 50000, 1.46385)
sV1 = soldiers(a[1][0], a[1][1], 50000, 1.46385)
uV2 = soldiers(a[0][0], a[0][1], 50000, -1.46385)
sV2 = soldiers(a[1][0], a[1][1], 50000, -1.46385)

M1 = foreMap(5.5, foreMap(5.5, uV1))
M2 = backMap(5.5, backMap(5.5, sV1))
M3 = foreMap(5.5, foreMap(5.5, uV2))
M4 = backMap(5.5, backMap(5.5, sV2))

N = []
M = []

for i in range(50000, 0, -1):
    if M1[i][1] < M4[i][1]:
        N.append(M1[i])
    else:
        break

for i in range(50000, 0, -1):
    if M4[i][1] > M1[i][1]:
        M.append(M4[i])
        flag = True
    else:
        break
for i in range(len(M) - 1, -1, -1):
    N.append(M[i])

M = []

for i in range(50000, 100000):
    if M3[i][1] > M2[i][1]:
        N.append(M3[i])
        flag = True
    else:
        break

for i in range(50000, 100000):
    if M2[i][1] < M3[i][1]:
        M.append(M2[i])
        flag = True
    else:
        break
for i in range(len(M) - 1, -1, -1):
    N.append(M[i])

p = 1.15108
q = 0.49643

A = [-1, 0, 1, 1.46385, -1.46385, p, -p, q, -q]
B = [-1, 0, 1, -1.46385, 1.46385, q, -q, p, -p]

x1 = []
x2 = []
y1 = []
y2 = []

x3 = []
x4 = []
y3 = []
y4 = []

x = []
y = []

Wu = foreMap(5.5, N)
Ws = backMap(5.5, N)

for i in M3:
    x1.append(i[0])
    y1.append(i[1])
for i in M4:
    x2.append(i[0])
    y2.append(i[1])
for i in Wu:
    x3.append(i[0])
    y3.append(i[1])
for i in Ws:
    x4.append(i[0])
    y4.append(i[1])
for i in N:
    x.append(i[0])
    y.append(i[1])

plt.figure(figsize=(10, 10))

plt.xlim(-3, 3)
plt.ylim(-3, 3)

# plt.plot(x1,y1, color='blue')
# plt.plot(x2,y2, color='red')
plt.plot(x3, y3, color='blue')
plt.plot(x4, y4, color='red')
plt.plot(x, y, 'black')

plt.scatter(A, B, s=10, c='black')

plt.show()
