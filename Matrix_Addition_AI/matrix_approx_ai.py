from math import pow
from Network import Network
from random import randint
from Target import Target
from time import sleep


def main():
    target = Target()
    network = Network()
    repeatForwardPass(0, 10000, target, network)
    network.clear_temps()

    repeatForwardPass(10000,10000,target, network)
    network.clear_temps()
    repeatForwardPass(20000,10000,target, network)
    network.clear_temps()
    repeatForwardPass(30000,10000,target, network)
    network.clear_temps()


def errorCalculation(target: list, network: Network):
    """
    Updates the error values for the current network to be the average error
    :param list target: the 4x1 target matrix of values
    :param Network network: The network to forward feed through
    """

    ## find total cost of forward propagation

    # cost = each element in aL minus corresponding element squared
    for i in range(0,4):
        network._error_total[i][0] = pow((network._activationL[i][0] - target[i][0]), 2)



def feedForwardPass(x: list, target: list, network: Network):
    """
    Single pass feeding forward through the network with the specified input
   :param list x: The input 8x1 matrix
   :param list target: the target training 4x1 matrix
   :param Network network: The network to forward feed through
   """
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

    # need to apply activation function

    for i in range(0,10):
        network._activation1[i][0] = activationFunction(network._z1[i][0])
             

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


    # need to apply activation function

    for i in range(0,8):
        network._activation2[i][0] = activationFunction(network._z2[i][0])

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

    # need to apply activation function

    for i in range(0,4):
        network._activationL[i][0] = activationFunction(network._z3[i][0])

    # print(network._activationL)
    # print(network._z3)



    errorCalculation(target, network)
    

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
    gradientDescentDerivatives(network)

    sleep(1)

def activationFunction(n: int):
    if (n <= 0):
        return 0
    else:
        return n



def gradientDescentDerivatives(network: Network):
    ## change in C wrt last layer
    ## times change in last layer wrt z

    delta_L = [[0.0],[0],[0],[0]]
    
    for i in range(0,4):
        delta_L[i][0] = network._activationL[i][0]

    delta_lminus1 = [[0],[0],[0],[0],[0],[0],[0],[0]]

    delta_lminus2 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

    for i in range(0,4):
        delta_L[i][0] = activationFunction(network._z3[i][0])

    ## every layer after that is more difficult

    ## need to transpose W and multiply by delta_L
    ## then after that multiply elementwise with the activation of the current node's z's
        
    ## repeat for delta_3
        
    ## how the cost is affected by the change in each bias
    dCdb3 = delta_L

    dCdb2 = delta_lminus1

    dCdb1 = delta_lminus2

    ## how the cost is affected by the change in each weight

    ## 10x8
    dCdW1 = [[0,0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    

  

    #8x10.0
    dCdW2 = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]

    #4x8
    dCdW3 = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
             [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    

    for i in range(0,4):
        ## iterate through all the rows
        ##use delta_L as the error for each of these repeats -> delta_L[i][0]
        ## use activation 2 as the transposed matrix -> normally a 8x1 matrix
        for j in range(0,8):
            dCdW3[i][j] = (delta_L[i][0] * network._activation2[j][0])

    # print(dCdW3, end="\n\n\n")




    ## now adjust the values based on the formula


    return 0


    

    



if __name__ == "__main__":
        main()
        