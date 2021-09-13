import numpy as np

def hw1() :
    Hw1_1 = np.random.randint(0, 10, size = (4, 5))
    print(Hw1_1)
    print("---------------")
    Hw1_2 = Hw1_1.T
    print(Hw1_2)
    print("---------------")
    Hw1_3 = Hw1_1 ** 2
    print(Hw1_3)
    print("---------------")
    Hw1_4 = Hw1_1 + Hw1_3
    print(Hw1_4)
    print("---------------")
    Hw1_5 = Hw1_2 @ Hw1_1
    print(Hw1_5)
    print("---------------")

hw1()