import logic_gate

#Creating an instance of the LogicGate class called 'lg'
lg = logic_gate.LogicGate()

#Test Cases
tests = [[0,0], [0,1], [1,0], [1, 1]]

print("AND gate:")
for test in tests:
    y = lg.and_gate(test[0], test[1])
    print(f'{test[0]} {test[1]} --> {y}')

print(" ")
print("NAND gate:")
for test in tests:
    y = lg.nand_gate(test[0], test[1])
    print(f'{test[0]} {test[1]} --> {y}')

print(" ")
print("OR gate:")
for test in tests:
    y = lg.or_gate(test[0], test[1])
    print(f'{test[0]} {test[1]} --> {y}')

print(" ")
print("NOR gate:")
for test in tests:
    y = lg.nor_gate(test[0], test[1])
    print(f'{test[0]} {test[1]} --> {y}')

print(" ")
print("XOR gate:")
for test in tests:
    y = lg.xor_gate(test[0], test[1])
    print(f'{test[0]} {test[1]} --> {y}')