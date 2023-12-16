from math import pow
from Network import Network
from random import randint
from Target import Target
from time import sleep


def main():
    target = Target()
    network = Network()
    repeatForwardPass(0, 10, target, network)

    repeatForwardPass(10,10,target, network)
    repeatForwardPass(20,10,target, network)
    repeatForwardPass(30,10,target, network)




def feedForwardPass(x: list, target: list, network: Network):

    # one pass of feed forward =>


    #compute z1
    #z1 = (W1 * x) + b1

    #matrix multiplication
    for i in range(0,10):

        sum = 0
        for j in range(0,8):

            sum = sum + x[j][0] * network._weights1[i][j]
                 
        sum = sum + network._bias1[i][0]

        network._z1[i][0] = sum

    # need to apply ReLu

    for i in range(0,10):
        
        if (network._z1[i][0] <= 0):
            network._activation1[i][0] = 0
        else:
            network._activation1[i][0] = network._z1[i][0]
             

    ## print the array out
        
    # print(network._activation1)
    # print(network._z1)


    #compute z2 and a2
            
    #z2 = (w2 * a1) + b2
    #w2 is 8x10


    for i in range(0,8):

        sum = 0
        for j in range(0,10):

            sum = sum + (network._activation1[j][0] * network._weights2[i][j])
                 
        sum = sum + network._bias2[i][0]

        network._z2[i][0] = sum


    # need to apply ReLu

    for i in range(0,8):
        
        if (network._z2[i][0] <= 0):
            network._activation2[i][0] = 0
        else:
            network._activation2[i][0] = network._z2[i][0]

    # print(network._activation2)
    # print(network._z2)
            

    ## final step in forwards propagation:
    # need to comupte aL
    #aL = (w3 * a2) + b3

    #w3 is a 4x8 matrix
    #a2 is 8x1
    for i in range(0,4):

        sum = 0
        for j in range(0,8):

            sum = sum + (network._activation2[j][0] * network._weights3[i][j])
                 
        sum = sum + network._bias3[i][0]

        network._z3[i][0] = sum

    # need to apply ReLu

    for i in range(0,4):
        
        if (network._z3[i][0] <= 0):
            network._activationL[i][0] = 0
        else:
            network._activationL[i][0] = network._z3[i][0]

    # print(network._activationL)
    # print(network._z3)



    ## find total cost of forward propagation

    # cost = each element in aL minus corresponding element squared
    for i in range(0,4):
        network._error_total[i][0] = pow((network._activationL[i][0] - target[i][0]), 2)

    print("Total error" + network._error_total.__str__())
    

    ## need to repeat m-times for a batch of entries




def repeatForwardPass(n: int, m: int, target: Target, network: Network):
    """
    Used to repeat the forward pass through the algorithm
   :param int n: the offset for the test number
   :param int m: The amount of passes in current cluster.
   :param list x: The input 8x1 matrix
   :param Target target: the target training set object containing the selected 8x1 matrix
   :param Network network: The network to forward feed through
   """
    

    ##  need to set x and the target array here on each iteration
   

    for k in range(0,m):
        
        xArr = [[target._inputs[0][n+ k]],
             [target._inputs[1][n + k]],
             [target._inputs[2][n + k]],
             [target._inputs[3][n + k]],
             [target._inputs[4][n + k]],
             [target._inputs[5][n + k]],
             [target._inputs[6][n + k]],
             [target._inputs[7][n + k]]]
        targetArr = [[target._targets[0][n + k]],
                  [target._targets[1][n + k]],
                  [target._targets[2][n + k]],
                  [target._targets[3][n + k]]]
        feedForwardPass(xArr, targetArr, network)


    for i in range(0,4):
        network._error_total[i][0] = network._error_total[i][0] / m

    print("Mean squared error for the sample: " + network._error_total.__str__())
    sleep(1)


    



if __name__ == "__main__":
        main()
        