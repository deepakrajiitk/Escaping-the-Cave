#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from pyfinite import ffield


# In[3]:

F = ffield.FField(7, gen=0x83, useLUT=-1)
exponential = [[-1] * 128 for i in range(128)]


def getAscii(ch):
    return ord(ch)


def fastExponentiation(a, n):
    if n == 0:
        return 1
    x = fastExponentiation(a, n // 2)
    x = F.Multiply(x, x)
    if n % 2 == 1:
        x = F.Multiply(x, a)
    return x


def doOperation(row, ele, res):
    temp = []
    for ele1, ele2 in zip([F.Multiply(e, ele) for e in row], res):
        temp.append((int(ele1) ^ int(ele2)))
    return temp


def LinearTransform(matrix, ele_list):
    res = [0] * 8
    for row, ele in zip(matrix, ele_list):
        res = doOperation(row, ele, res)
    return res


# In[4]:


def findCipherPair(cipher):
    temp = [cipher[i : i + 2] for i in range(0, len(cipher), 2)]
    return temp


def decode_block(cipher):
    cipher_pairs = findCipherPair(cipher)
    plain = ""
    for x in cipher_pairs:
        plain += chr(
            16 * (getAscii(x[0]) - getAscii("f")) + getAscii(x[1]) - getAscii("f")
        )
    return plain


# In[5]:
def possExponsfun():
    return [[] for i in range(8)]


def possDiagsfun():
    return [[[] for i in range(8)] for _ in range(8)]


def compute(possE, possD, inS, outS, i, j):
    flag = True
    for inp, out in zip(inS, outS):
        if ord(out) != fastExponentiation(
            F.Multiply(
                fastExponentiation(F.Multiply(fastExponentiation(ord(inp), i), j), i), j
            ),
            i,
        ):
            flag = False
            break
    if flag:
        possE[idx].append(i)
        possD[idx][idx].append(j)


# for diagonal elements
possibleExp = possExponsfun()
possibleDiagVals = possDiagsfun()

plaintext = open("plaintext_file.txt", "r")
ciphertext = open("ciphertext_file.txt", "r")

for idx, (inp, out) in enumerate(zip(plaintext.readlines(), ciphertext.readlines())):
    input_string = [decode_block(msg)[idx] for msg in inp.strip().split(" ")]
    output_string = [decode_block(msg)[idx] for msg in out.strip().split(" ")]

    for i in range(1, 127):
        for j in range(1, 128):
            compute(possibleExp, possibleDiagVals, input_string, output_string, i, j)
# print("Possible diagonal values: ")
# print(possibleDiagVals)
# print("Possible exponents: ")
# print(possibleExp)


# In[9]:
def expectedValue(inp, p1, e1, p2, e2, i):
    temp1 = fastExponentiation(
        int(
            F.Multiply(
                fastExponentiation(
                    F.Multiply(fastExponentiation(getAscii(inp), p2), e2), p2
                ),
                i,
            )
        )
        ^ int(
            F.Multiply(
                fastExponentiation(
                    F.Multiply(fastExponentiation(getAscii(inp), p2), i), p1
                ),
                e1,
            )
        ),
        p1,
    )
    return temp1


def evaluate(possibleE, possD, ind, p1, e1, p2, e2, i):
    flag = True
    for inp, outp in zip(inpString, outString):
        if ord(outp) != expectedValue(inp, p1, e1, p2, e2, i):
            flag = False
            break
    if flag:
        possibleE[ind + 1] = [p1]
        possD[ind + 1][ind + 1] = [e1]
        possibleE[ind] = [p2]
        possD[ind][ind] = [e2]
        possD[ind][ind + 1] = [i]


plaintext = open("plaintext_file.txt", "r")
ciphertext = open("ciphertext_file.txt", "r")

for ind, (iline, oline) in enumerate(
    zip(plaintext.readlines(), ciphertext.readlines())
):
    if ind > 6:
        break

    inpString = [decode_block(msg)[ind] for msg in iline.strip().split(" ")]
    outString = [decode_block(msg)[ind + 1] for msg in oline.strip().split(" ")]

    for i in range(1, 128):
        for p1, e1 in zip(possibleExp[ind + 1], possibleDiagVals[ind + 1][ind + 1]):
            for p2, e2 in zip(possibleExp[ind], possibleDiagVals[ind][ind]):
                evaluate(possibleExp, possibleDiagVals, ind, p1, e1, p2, e2, i)
# print(possibleDiagVals)
# print(possibleExp)


# In[18]:
def findPlainOutput(plaintext):
    plain = []
    for c in plaintext:
        plain.append(getAscii(c))
    output = [[0] * 8 for i in range(8)]
    return plain, output


def EAEAE(plaintext, lin_mat, exp_mat):  # Defines EAEAE
    plain, out = findPlainOutput(plaintext)

    for index, ele in enumerate(plain):
        out[0][index] = fastExponentiation(ele, exp_mat[index])
    out[1] = LinearTransform(lin_mat, out[0])
    for index, ele in enumerate(out[1]):
        out[2][index] = fastExponentiation(ele, exp_mat[index])
    out[3] = LinearTransform(lin_mat, out[2])
    for index, ele in enumerate(out[3]):
        out[4][index] = fastExponentiation(ele, exp_mat[index])
    return out[4]


def generateString(temp):
    inp_string = []
    for msg in temp.strip().split(" "):
        inp_string.append(decode_block(msg))
    return inp_string


def find(lin_trans_list, exp_list, i):
    flag = True
    for inps, outs in zip(inp_string, out_string):
        if EAEAE(inps, lin_trans_list, exp_list)[idx + off] != getAscii(
            outs[idx + off]
        ):
            flag = False
            break
    if flag == True:
        possibleDiagVals[idx][idx + off] = [i]


for idx in range(0, 6):
    off = idx + 2
    exp_list = [e[0] for e in possibleExp]
    lin_trans_list = []
    for j in range(8):
        temp = []
        lin_trans_list.append([0 for i in range(8)])

    for i in range(8):
        lin_trans_list[i] = [
            possibleDiagVals[i][j][0] if len(possibleDiagVals[i][j]) != 0 else 0
            for j in range(8)
        ]

    plaintext = open("plaintext_file.txt", "r")
    ciphertext = open("ciphertext_file.txt", "r")

    for idx, (inp, out) in enumerate(
        zip(plaintext.readlines(), ciphertext.readlines())
    ):
        if idx > (7 - off):
            continue

        inp_string = generateString(inp)
        out_string = generateString(out)

        for i in range(1, 128):
            lin_trans_list[idx][idx + off] = i
            find(lin_trans_list, exp_list, i)

lin_trans_list = []
for j in range(8):
    lin_trans_list.append([0 for i in range(8)])

for i in range(0, 8):
    lin_trans_list[i] = [
        possibleDiagVals[i][j][0] if len(possibleDiagVals[i][j]) != 0 else 0
        for j in range(8)
    ]

# print("====================")
# print(lin_trans_list)
# print(exp_list)
# print("====================")


# In[19]:


# Final E and A as found above
At = [
    [84, 115, 12, 125, 97, 25, 21, 66],
    [0, 70, 29, 16, 33, 51, 120, 13],
    [0, 0, 43, 2, 1, 31, 22, 85],
    [0, 0, 0, 12, 119, 47, 100, 27],
    [0, 0, 0, 0, 112, 110, 2, 21],
    [0, 0, 0, 0, 0, 11, 88, 66],
    [0, 0, 0, 0, 0, 0, 27, 31],
    [0, 0, 0, 0, 0, 0, 0, 38],
]
E = [19, 116, 41, 80, 92, 42, 22, 21]
# print("E vector: ", E)
A = np.array([[At[j][i] for j in range(len(At))] for i in range(len(At[0]))])
# print("A matrix: ", A)


# In[24]:
def calcExponents(expos):
    for b in range(0, 128):
        for e in range(1, 127):
            expos[b] += [F.Multiply(expos[b][e - 1], b)]


def calcEinv(expos, einv):
    for b in range(0, 128):
        for e in range(1, 127):
            einv[e][expos[b][e]] = b


def calcAug(aug, block_size):
    for i in range(0, block_size):
        for j in range(0, block_size):
            aug[i][j] = A[i][j]
        aug[i][i + j + 1] = 1


def calcInv(inv, aug, block_size):
    for i in range(0, block_size):
        for j in range(0, block_size):
            inv[i][j] = aug[i][block_size + j]


def iniit(a, b):
    return np.array([[0] * a for _ in range(b)])


def findInverses():
    temp = [1]
    for i in range(1, 128):
        temp.extend([F.Inverse(i)])
    return temp


block_size = 8
augA = iniit(16, 8)
invA = iniit(8, 8)
Einv = iniit(128, 128)

exponents = [[1] for i in range(128)]
calcExponents(exponents)
calcEinv(exponents, Einv)

inverses = findInverses()

calcAug(augA, block_size)

for j in range(0, block_size):
    assert np.any(augA[j:, j] != 0)
    pivot_row = np.where(augA[j:, j] != 0)[0][0] + j
    augA[[j, pivot_row]] = augA[[pivot_row, j]]
    # make pivot k 1
    mul_fact = inverses[augA[j][j]]
    for k in range(block_size * 2):
        augA[j][k] = F.Multiply(augA[j][k], mul_fact)
    for i in range(0, block_size):
        if i != j and augA[i][j] != 0:
            mult_fact = augA[i][j]
            for k in range(block_size * 2):
                temp = F.Multiply(augA[j][k], mult_fact)
                augA[i][k] = F.Add(temp, augA[i][k])

calcInv(invA, augA, block_size)

# print("A inverse matrix: \n{}".format(invA))


# In[27]:


password = "gskohggkhqglmhlumrmnjhlohlimjtgm"  # Encrypted password
password = ["gskohggkhqglmhlu", "mrmnjhlohlimjtgm"]


def findEinverse(block, E):
    transformed = [Einv[E[j]][block[j]] for j in range(8)]
    return transformed


def findAinverse(block, A):
    transformed = [0] * 8
    for row in range(0, 8):
        elem_sum = 0
        for col in range(0, 8):
            elem_sum = F.Add(F.Multiply(A[row][col], block[col]), elem_sum)
        transformed[row] = elem_sum
    return transformed


final_password = ""
for i in range(0, 2):
    elements = password[i]
    currentBlock = [
        (getAscii(elements[2 * j]) - getAscii("f")) * 16
        + (getAscii(elements[2 * j + 1]) - getAscii("f"))
        for j in range(8)
    ]

    for ch in findEinverse(
        findAinverse(
            findEinverse(findAinverse(findEinverse(currentBlock, E), invA), E), invA
        ),
        E,
    ):
        final_password += chr(ch)
print("The password is", final_password)

# %%
