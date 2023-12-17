from random import random

class Target:

    def __init__(self):
        self._inputs = [[],
                          [],
                          [],
                          [],
                          [],
                          [],
                          [],
                          []]
        self._targets = [[],
                         [],
                         [],
                         []]

        ## make 100000 test samples
        for v in range(0,100000):



            for i in range(0,8):
                self._inputs[i].append((random() * 20) - 10)

            ## of the form [current row][v]
            self._targets[0].append(self._inputs[0][v] + self._inputs[4][v])
            self._targets[1].append(self._inputs[1][v] + self._inputs[5][v])
            self._targets[2].append(self._inputs[2][v] + self._inputs[6][v])
            self._targets[3].append(self._inputs[3][v] + self._inputs[7][v])

        # print("Inputs: " + self._inputs.__str__())
        # print("Targets: " + self._targets.__str__())
