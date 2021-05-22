#!/usr/bin/env python

from voting import _tmp as tmp


if __name__ == "__main__":

    AN = []
    path = 'experiments/ijcai/paths+urn+phi_mallows+preflib/controllers/distances/antagonism.txt'
    with open(path, 'r') as file_txt:
        for i in range(494):
            line = float(file_txt.readline().strip())
            AN.append(line)

    ST = []
    path = 'experiments/ijcai/paths+urn+phi_mallows+preflib/controllers/distances/stratification.txt'
    with open(path, 'r') as file_txt:
        for i in range(494):
            line = float(file_txt.readline().strip())
            ST.append(line)

    ID = []
    path = 'experiments/ijcai/paths+urn+phi_mallows+preflib/controllers/distances/identity.txt'
    with open(path, 'r') as file_txt:
        for i in range(494):
            line = float(file_txt.readline().strip())
            ID.append(line)

    UN = []
    path = 'experiments/ijcai/paths+urn+phi_mallows+preflib/controllers/distances/uniformity.txt'
    with open(path, 'r') as file_txt:
        for i in range(494):
            line = float(file_txt.readline().strip())
            UN.append(line)

    an = 0
    st = 0
    remis = 0
    for i in range(329, 494):
        if AN[i] < ST[i]:
            an += 1
        elif AN[i] > ST[i]:
            st += 1
        else:
            remis += 1

    print('AN', an, ' -- ST', st)
    print('draw stan', remis)

    un = 0
    id = 0
    remis = 0
    for i in range(329, 494):
        if UN[i] < ID[i]:
            un += 1
        elif UN[i] > ID[i]:
            id += 1
        else:
            remis += 1

    print('un', un, ' -- id', id)
    print('draw idun', remis)

    print('ST', sum(ST[329:494])/(494-329))
    print('AN', sum(AN[329:494])/(494-329))

