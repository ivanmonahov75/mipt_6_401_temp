import numpy as np

num_exp = int(input("Enter number of experiments for equal ion power "))
volume = float(input("Enter volume of one batch, ml "))

print("Volume of 0.3 NaOH, ml:", (np.linspace(0.3, 0.05, num_exp)*volume/0.3))
print("Volume of 0.3 NaCl, ml:", (np.linspace(0.3, 0.3, num_exp)*volume/0.3 - np.linspace(0.3, 0.05, num_exp)*volume/0.3))


num_exp = int(input("Enter number of experiments for differnt ion power "))
volume = float(input("Enter volume of one batch, ml "))

print("Volume of Naoh, ml:", np.linspace(volume/3, volume/3, num_exp))
print("Volume of NaCl 2.4M, ml:", (np.linspace(0.1, 1.7, num_exp) - 0.1)*volume/2.4)
print("Volume of H2O, ml:", np.linspace(volume/3, volume/3, num_exp)*2 - (np.linspace(0.1, 1.7, num_exp) - 0.1)*volume/2.4)


