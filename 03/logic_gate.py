import numpy as np

#Logic gate class
class LogicGate():
    def __init__(self):
        pass
    
    #Function for AND gate
    def and_gate(self, x1, x2):
        b = -0.7
        w = np.array([0.5, 0.5, 1])
        x = np.array([x1, x2, b])

        y = np.sum(w * x)

        if y > 0:
            return 1
        else:
            return 0

    #Function for NAND gate
    def nand_gate(self, x1, x2):
        b = -0.7
        w = np.array([0.5, 0.5, 1])
        x = np.array([x1, x2, b])

        y = np.sum(w * x)

        if y > 0:
            return 0
        else:
            return 1
    
    #Function for OR gate
    def or_gate(self, x1, x2):
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])

        y = np.sum(w * x)

        if y > 0:
            return 1
        else:
            return 0
    
    #Function for NOR gate
    def nor_gate(self, x1, x2):
        b = -0.9
        w = np.array([1, 1, 1])
        x = np.array([x1, x2, b])

        y = np.sum(w * x)

        if y > 0:
            return 0
        else:
            return 1
    
    #Function for XOR gate
    def xor_gate(self, x1, x2):
        y1 = self.or_gate(x1, x2)
        y2 = self.nand_gate(x1, x2)
        return self.and_gate(y1, y2)

if __name__ == '__main__':
    print("Logic Gate Class")