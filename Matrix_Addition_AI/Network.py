
class Network:

    def __init__(self):

        ## 10x8 matrix of weights from layer 0 to 1
        self._weights1 = [[4.0, -5.0, 2.0, 6.0, 7.0, 3.0, -6.0, -8.0],
                            [-3.0, -7.0, -3.0, 5.0, -8.0, 4.0, 6.0, -3.0],
                            [6.0, 8.0, 7.0, 4.0, 10.0, 1.0, 2.0, 3.0],
                            [1.0, -4.0, 9.0, 10.0, 2.0, 7.0, 6.0, 5.0],
                            [4.0, 5.0, 2.0, 6.0, 7.0, 3.0, 6.0, 8.0],
                            [3.0, 7.0, -3.0, 5.0, -8.0, -4.0, 6.0, 3.0],
                            [6.0, -8.0, 7.0, 4.0, 10.0, 1.0, 2.0, 3.0],
                            [1.0, -4.0, 9.0,  10.0, 2.0, 7.0, -6.0, 5.0],
                            [-6.0, 8.0, 7.0, 4.0, -10.0, 1.0, 2.0, 3.0],
                            [1.0, 4.0, 9.0,  10.0, 2.0, 7.0, 6.0, -5.0]]
        
        ## The bias vector applied to the first layer
        self._bias1 = [[3.0],[-4.0],[-2.0],[-8.0],[4.0],[9.0],[1.0],[3.0],[5.0],[3.0]]

        ## The activation/output of the first layer
        self._activation1 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

        ## The resultant for layer 1 before applying activation function
        self._z1 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

        ## 8x10 matrix of weights from layer 1 to 2
        self._weights2 = [[4.0, -10.0, 9.0, 3.0, 7.0, 8.0, 2.0, -6.0, 5.0, 1.0],
          [4.0, 6.0, -1.0, 5.0, 8.0, 2.0, 7.0, 3.0, 10.0, 9.0],
          [4.0, 6.0, 1.0, 5.0, 8.0, 2.0, 7.0, 3.0, 10.0, 9.0],
          [4.0, 10.0, 9.0, 3.0, 7.0, -8.0, 2.0, -6.0, 5.0, 1.0],
          [4.0, -10.0, 9.0, 3.0, 7.0, 8.0, 2.0, 6.0, 5.0, 1.0],
          [2.0, -7.0, -8.0, -4.0, -10.0, 3.0, 6.0, -5.0, 9.0, 1.0],
          [4.0, 6.0, 1.0, 5.0, -8.0, 2.0, 7.0, 3.0, 10.0, 9.0],
          [7.0, -2.0, 9.0, 4.0, -3.0, 6.0, 10.0, 5.0, -1.0, 8.0]]
        
        ## The bias vector applied to the first layer
        self._bias2 = [[3.0],[-4.0],[2.0],[8.0],[-4.0],[-9.0],[1.0],[3.0]]

        ## The activation/output of the second layer
        self._activation2 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

        ## The resultant for layer 2 before applying activation function
        self._z2 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]

        ## 4x8 matrix of weights from layer 2 to output layer
        self._weights3 = [[4.0, -5.0, 2.0, 6.0, 7.0, 3.0, 6.0, 8.0],
          [3.0, 7.0, 3.0, 5.0, -8.0, 4.0, 6.0, 3.0],
          [6.0, 8.0, 7.0, -4.0, 10.0, 1.0, 2.0, -3.0],
          [1.0, 4.0, 9.0,  10.0, 2.0, 7.0, 6.0, -5.0]]
        
        ## The bias vector applied to the third layer
        self._bias3 = [[-3.0],[2.0],[-15.0],[-10.0]]

        ## The activation/output of the third layer
        self._activationL = [[0.0],[0.0],[0.0],[0.0]]

        ## The resultant for layer 2 before applying activation function
        self._z3 = [[0.0],[0.0],[0.0],[0.0]]

        ## total error vector for single instance
        self._error_total = [[0.0],[0.0],[0.0],[0.0]]

    # def __init__(x: list, self):
    #     self.__init__(self)
    #     self._inputX = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
    #     for i in range(0,8):
    #         self._inputX[i][0] = x[i][0]



        


    def clear_temps(self):
        self._activation1 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        self._z1 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        self._activation2 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        self._z2 = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        self._activationL = [[0.0],[0.0],[0.0],[0.0]]
        self._z3 = [[0.0],[0.0],[0.0],[0.0]]
        self._error_total = [[0.0],[0.0],[0.0],[0.0]]


if __name__ == "__main__":
    ## testing the Network class
    newNetwork = Network()

    # print(newNetwork._error_total)

