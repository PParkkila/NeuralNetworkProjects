from math import pow





def network(x, target):
        
    print("code runs...\n\n")


    #initializing all matrices

    # 8x1 matrix for input - x

    

    #weight 1 is a 10x8 matrix - w1
    w1 = [[4, -5, 2, 6, 7, 3, -6, -8],
          [-3, -7, -3, 5, -8, 4, 6, -3],
          [6, 8, 7, 4, 10, 1, 2, 3],
          [1, -4, 9,  10, 2, 7, 6, 5],
          [4, 5, 2, 6, 7, 3, 6, 8],
          [3, 7, -3, 5, -8, -4, 6, 3],
          [6, -8, 7, 4, 10, 1, 2, 3],
          [1, -4, 9,  10, 2, 7, -6, 5],
          [-6, 8, 7, 4, -10, 1, 2, 3],
          [1, 4, 9,  10, 2, 7, 6, -5]]
    #b1 is a 10x1 matrix
    b1 = [[3],[-4],[-2],[-8],[4],[9],[1],[3],[5],[3]]
    #a1 is a 10x1 matrix
    a1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    z1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

    #weight 2 is a 8x10 matrix
    w2 = [[4, -10, 9, 3, 7, 8, 2, -6, 5, 1],
          [4, 6, -1, 5, 8, 2, 7, 3, 10, 9],
          [4, 6, 1, 5, 8, 2, 7, 3, 10, 9],
          [4, 10, 9, 3, 7, -8, 2, -6, 5, 1],
          [4, -10, 9, 3, 7, 8, 2, 6, 5, 1],
          [2, -7, -8, -4, -10, 3, 6, -5, 9, 1],
          [4, 6, 1, 5, -8, 2, 7, 3, 10, 9],
          [7, -2, 9, 4, -3, 6, 10, 5, -1, 8]]
    #b2 is a 8x1 matrix
    b2 = [[3],[-4],[2],[8],[-4],[-9],[1],[3]]
    #a2 is a 8x1 matrix
    a2 = [[0],[0],[0],[0],[0],[0],[0],[0]]
    z2 = [[0],[0],[0],[0],[0],[0],[0],[0]]

    #weight 3 is a 4x8 matrix
    w3 = [[4, -5, 2, 6, 7, 3, 6, 8],
          [3, 7, 3, 5, -8, 4, 6, 3],
          [6, 8, 7, -4, 10, 1, 2, -3],
          [1, 4, 9,  10, 2, 7, 6, -5]]

    #aL is a 4x1 matrix
    b3 = [[-3],[2],[-15],[-10]]
    aL = [[0],[0],[0],[0]]
    z3 = [[0],[0],[0],[0]]

    error_total = [[0], [0], [0], [0]]


    # one pass of feed forward =>


    #compute z1
    #z1 = (W1 * x) + b1

    #matrix multiplication
    for i in range(0,10):

        sum = 0
        for j in range(0,8):

            sum = sum + x[j][0] * w1[i][j]
                 
        sum = sum + b1[i][0]

        z1[i][0] = sum

    # need to apply ReLu

    for i in range(0,10):
        
        if (z1[i][0] <= 0):
            a1[i][0] = 0
        else:
            a1[i][0] = z1[i][0]
             

    ## print the array out
        
    print(a1)
    print(z1)


    #compute z2 and a2
            
    #z2 = (w2 * a1) + b2
    #w2 is 8x10


    for i in range(0,8):

        sum = 0
        for j in range(0,10):

            sum = sum + (a1[j][0] * w2[i][j])
                 
        sum = sum + b2[i][0]

        z2[i][0] = sum


    # need to apply ReLu

    for i in range(0,8):
        
        if (z2[i][0] <= 0):
            a2[i][0] = 0
        else:
            a2[i][0] = z2[i][0]

    print(a2)
    print(z2)
            

    ## final step in forwards propagation:
    # need to comupte aL
    #aL = (w3 * a2) + b3

    #w3 is a 4x8 matrix
    #a2 is 8x1
    for i in range(0,4):

        sum = 0
        for j in range(0,8):

            sum = sum + (a2[j][0] * w3[i][j])
                 
        sum = sum + b3[i][0]

        z3[i][0] = sum

    # need to apply ReLu

    for i in range(0,4):
        
        if (z3[i][0] <= 0):
            aL[i][0] = 0
        else:
            aL[i][0] = z3[i][0]

    print(aL)
    print(z3)



    ## find total cost of forward propagation

    # cost = each element in aL minus corresponding element squared
    for i in range(0,4):
        error_total[i][0] = pow((aL[i][0] - target[i][0]), 2)

    print("Total error" + error_total.__str__())
    

    ## need to repeat m-times for a batch of entries














if __name__ == "__main__":
        a = 1.0
        b = 2.0
        c = 3.0
        d = 4.0

        e = 1.0
        f = 2.0
        g = 3.0
        h = 4.0


        t1 = a + e
        t2 = b + f
        t3 = c + g
        t4 = d + h


        target = [[t1],[t2], [t3], [t4]]


        # a, b, c, d , e, f, g, h
        x = [[a],[b],[c],[d],[e],[f],[g],[h]]
        network(x, target)