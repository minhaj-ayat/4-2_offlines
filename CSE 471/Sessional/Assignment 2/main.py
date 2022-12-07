import math
import numpy as np
import scipy.stats

Y = {}
P = [[0.0, 0.0], [0.0, 0.0]]
Nu = {}
Ns = {}
States = 2
ip = []


def readInput():
    global Y, P, States, Ns, Nu, ip
    file1 = open("IO/Inp/data.txt")
    lines = file1.readlines()

    ii = 1
    for ln in lines:
        Y[ii] = float(ln)
        ii += 1
    # print(Y)

    file2 = open("IO/Inp/parameters.txt")
    fl = file2.readline()
    States = int(fl)
    print(States)
    for k in range(0, States):
        fl = file2.readline()
        fl = fl.split()
        c = 0
        for el in fl:
            P[k][c] = float(el)
            c += 1
    fl = file2.readline().split()
    c = 1
    for el in fl:
        Nu[c] = float(el)
        c += 1
    fl = file2.readline().split()

    c = 1
    for el in fl:
        Ns[c] = float(el)
        c += 1

    print(P)
    print(Nu)
    print(Ns)


def normal_dist(x, mean, sd):
    return (1 / (sd * math.sqrt(2 * 3.1416))) * np.exp((-0.5) * ((x - mean) / sd) ** 2)


def viterbi(sig):
    global ip
    Pc = np.array(P).T.tolist()
    Cc = [0 for x in range(0, States)]
    Cc[States - 1] = 1

    for i in range(0, States):
        for j in range(0, States):
            if i == j:
                Pc[i][j] -= 1

    for k in range(0, States):
        Pc[States - 1][k] = 1
    print(Pc)
    print(Cc)

    ip = np.linalg.solve(Pc, Cc)

    S = [[0 for i in range(len(Y))] for j in range(States)]
    for k in range(0, States):
        S[k][0] = np.log(ip[k] * scipy.stats.norm.pdf(Y[1], Nu[k + 1], math.sqrt(Ns[k + 1])))

    parents = {}
    for i in range(1, len(Y)):
        ind = -1
        for k in range(0, States):
            tl = [k, i]
            ts = str(tl)
            mx = np.NINF
            for l in range(0, States):
                val = (S[l][i - 1]) + np.log(P[l][k]) + np.log(
                    (scipy.stats.norm.pdf(Y[i + 1], Nu[k + 1], math.sqrt(Ns[k + 1]))))
                if val > mx:
                    mx = val
                    ind = l
            S[k][i] = mx
            parents[ts] = [ind, i - 1]

    count0 = 0
    writefile = []

    lmx = np.NINF
    fi = []
    fs = ""
    ms = -1
    for k in range(States):
        if S[k][len(Y) - 1] > lmx:
            lmx = S[k][len(Y) - 1]
            fi = [k, len(Y) - 1]
            fs = str(fi)
            ms = k
    writefile.append(ms)
    for i in range(len(Y) - 1):
        c = parents[fs]
        writefile.append(c[0])
        fs = str(c)
    writefile.reverse()

    if sig == 1:
        wf = open("viterbi.txt", 'w')
        for el in writefile:
            if el == 0:
                count0 += 1
                ws = "El Nino"
            else:
                ws = "La Nina"
            wf.write("\"" + ws + "\"\n")
        # print(np.array(S).T.tolist())
        print(count0)
    else:
        wf = open("viterbi_learned.txt", 'w')
        for el in writefile:
            if el == 0:
                count0 += 1
                ws = "El Nino"
            else:
                ws = "La Nina"
            wf.write("\"" + ws + "\"\n")
        # print(np.array(S).T.tolist())
        print(count0)


counter = 1
forward = [[0.0 for i in range(1001)] for j in range(States)]
backward = [[0.0 for p in range(1001)] for q in range(States)]


def baum_welch():
    global counter, forward, backward, Y, P
    if counter == 1:
        for k in range(0, States):
            forward[k][0] = ip[k]
        for k in range(0, States):
            backward[k][len(Y)] = 1 / States
        counter += 1

    for i in range(1, len(Y) + 1):
        s = 0
        for k in range(0, States):
            forward[k][i] = 0
            for l in range(0, States):
                forward[k][i] += forward[l][i - 1] * P[l][k]
            forward[k][i] *= scipy.stats.norm.pdf(Y[i], Nu[k + 1], math.sqrt(Ns[k + 1]))
            s += forward[k][i]
        for k in range(States):
            forward[k][i] /= s

    '''for i in range(len(Y) - 1, -1, -1):
        for k in range(0, States):
            backward[k][i + 1] = backward[k][i + 1] * scipy.stats.norm.pdf(Y[i + 1], Nu[k + 1], math.sqrt(Ns[k + 1]))
        s = 0
        for k in range(States):
            val = 0
            for l in range(States):
                val += backward[l][i + 1] * P[k][l]
            backward[k][i] = val
            s += backward[k][i]
        for k in range(States):
            backward[k][i] /= s'''
    for i in range(len(Y) - 1, -1, -1):
        s = 0
        for k in range(States):
            backward[k][i] = sum(P[k][ll] * scipy.stats.norm.pdf(Y[i+1], Nu[ll+1], math.sqrt(Ns[ll+1])) * backward[ll][i+1] for ll in range(States))
            s += backward[k][i]
        for k in range(States):
            backward[k][i] /= s

    fsink = 0
    for ll in range(States):
        fsink += forward[ll][len(Y)]

    pi1 = [[0.0 for i in range(len(Y)+1)] for j in range(States)]
    for i in range(len(Y)):
        s = 0
        for k in range(States):
            pi1[k][i] = (forward[k][i+1] * backward[k][i+1]) / fsink
            s += pi1[k][i]
        for k in range(States):
            pi1[k][i] /= s
    if counter == 2:
        print(np.array(pi1).T.tolist())

    pi2 = [[[0.0 for k in range(States)] for j in range(States)] for i in range(len(Y))]
    for i in range(len(Y)):
        s = 0
        for l in range(States):
            for k in range(States):
                pi2[i][l][k] = \
                    (forward[k][i] * P[l][k] * scipy.stats.norm.pdf(Y[i+1], Nu[l+1], math.sqrt(Ns[l+1])) * backward[l][i]) / fsink
                s += pi2[i][l][k]
        for l in range(States):
            for k in range(States):
                pi2[i][l][k] /= s
    if counter == 2:
        print("pi2", pi2)

    for k in range(States):
        s = 0
        for l in range(States):
            P[k][l] = sum(pi2[i][l][k] for i in range(len(Y)-1))
            s += P[k][l]
        for l in range(States):
            P[k][l] /= s
    if counter == 2:
        print(P)

    for k in range(States):
        s1 = 0
        s2 = 0
        for i in range(len(Y)):
            s1 += pi1[k][i] * Y[i+1]
            s2 += pi1[k][i]
        Nu[k+1] = s1/s2
    if counter == 2:
        print(Nu)

    for k in range(States):
        s1 = 0
        s2 = 0
        for i in range(len(Y)):
            s1 += pi1[k][i] * (Y[i + 1] - Nu[k+1]) * (Y[i + 1] - Nu[k+1])
            s2 += pi1[k][i]
        Ns[k + 1] = (s1 / s2)
    if counter == 2:
        print(Ns)

    print("Transition", P)
    print("Mean", Nu)
    print("Variance", Ns)
    '''print(np.array(forward).T.tolist())
    print(np.array(forward).shape)
    print(np.array(backward).T.tolist())
    print(np.array(backward).shape)'''
    counter += 1


readInput()
viterbi(1)
for c in range(8):
    baum_welch()

Pc = np.array(P).T.tolist()
Cc = [0 for x in range(0, States)]
Cc[States - 1] = 1

for i in range(0, States):
    for j in range(0, States):
        if i == j:
            Pc[i][j] -= 1

for k in range(0, States):
    Pc[States - 1][k] = 1
print(Pc)
print(Cc)

ip = np.linalg.solve(Pc, Cc)

pfile = [States]
for k in range(States):
    for l in range(States):
        pfile.append(P[k][l])
for k in range(States):
    pfile.append(Nu[k+1])
for k in range(States):
    pfile.append(Ns[k+1])
for k in range(States):
    pfile.append(ip[k])
wf = open("p_learned.txt", 'w')
cn = 0
for el in pfile:
    wf.write(str(el)+"\n")
viterbi(2)


