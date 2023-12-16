
class Network:

    def __init__(self):

        ## 10x8 matrix of weights from layer 0 to 1
        self._weights1 = [[4, -5, 2, 6, 7, 3, -6, -8],
                            [-3, -7, -3, 5, -8, 4, 6, -3],
                            [6, 8, 7, 4, 10, 1, 2, 3],
                            [1, -4, 9,  10, 2, 7, 6, 5],
                            [4, 5, 2, 6, 7, 3, 6, 8],
                            [3, 7, -3, 5, -8, -4, 6, 3],
                            [6, -8, 7, 4, 10, 1, 2, 3],
                            [1, -4, 9,  10, 2, 7, -6, 5],
                            [-6, 8, 7, 4, -10, 1, 2, 3],
                            [1, 4, 9,  10, 2, 7, 6, -5]]
        
        ## The bias vector applied to the first layer
        self._bias1 = [[3],[-4],[-2],[-8],[4],[9],[1],[3],[5],[3]]

        ## The activation/output of the first layer
        self._activation1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

        ## The resultant for layer 1 before applying activation function
        self._z1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

        ## 8x10 matrix of weights from layer 1 to 2
        self._weights2 = [[4, -10, 9, 3, 7, 8, 2, -6, 5, 1],
          [4, 6, -1, 5, 8, 2, 7, 3, 10, 9],
          [4, 6, 1, 5, 8, 2, 7, 3, 10, 9],
          [4, 10, 9, 3, 7, -8, 2, -6, 5, 1],
          [4, -10, 9, 3, 7, 8, 2, 6, 5, 1],
          [2, -7, -8, -4, -10, 3, 6, -5, 9, 1],
          [4, 6, 1, 5, -8, 2, 7, 3, 10, 9],
          [7, -2, 9, 4, -3, 6, 10, 5, -1, 8]]
        
        ## The bias vector applied to the first layer
        self._bias2 = [[3],[-4],[2],[8],[-4],[-9],[1],[3]]

        ## The activation/output of the second layer
        self._activation2 = [[0],[0],[0],[0],[0],[0],[0],[0]]

        ## The resultant for layer 2 before applying activation function
        self._z2 = [[0],[0],[0],[0],[0],[0],[0],[0]]

        ## 4x8 matrix of weights from layer 2 to output layer
        self._weights3 = [[4, -5, 2, 6, 7, 3, 6, 8],
          [3, 7, 3, 5, -8, 4, 6, 3],
          [6, 8, 7, -4, 10, 1, 2, -3],
          [1, 4, 9,  10, 2, 7, 6, -5]]
        
        ## The bias vector applied to the third layer
        self._bias3 = [[-3],[2],[-15],[-10]]

        ## The activation/output of the third layer
        self._activationL = [[0],[0],[0],[0]]

        ## The resultant for layer 2 before applying activation function
        self._z3 = [[0],[0],[0],[0]]

        ## total error vector for single instance
        self._error_total = [[0], [0], [0], [0]]


    def clear_temps(self):
        self._activation1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        self._z1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        self._activation2 = [[0],[0],[0],[0],[0],[0],[0],[0]]
        self._z2 = [[0],[0],[0],[0],[0],[0],[0],[0]]
        self._activationL = [[0],[0],[0],[0]]
        self._z3 = [[0],[0],[0],[0]]
        self._error_total = [[0], [0], [0], [0]]






if __name__ == "__main__":
    ## testing the Network class
    newNetwork = Network()

    # print(newNetwork._error_total)

