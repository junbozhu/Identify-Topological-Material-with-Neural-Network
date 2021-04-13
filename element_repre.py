import numpy as np

def element_repre(symbol_table, group_table, symbol_input_str): # input like 'Na'
    # input element symbol, output its atomic number, group number, period number
    if symbol_input_str in symbol_table:
        index = symbol_table.index(symbol_input_str)
        atomic_num = index + 1
        group_num = int(group_table[index])
    period_num = 0
    if atomic_num <=2:
        period_num = 1
    elif  3 <= atomic_num <= 10:
        period_num = 2
    elif  11 <= atomic_num <= 18:
        period_num = 3
    elif  19 <= atomic_num <= 36:
        period_num = 4
    elif  37 <= atomic_num <= 54:
        period_num = 5
    elif  55 <= atomic_num <= 86:
        period_num = 6
    elif  87 <= atomic_num <= 109:
        period_num = 7
    return (atomic_num, group_num, period_num ) # all number integer
    
    
A,B = np.loadtxt("ref_periodic_table.txt", unpack = True,dtype=str)
symbol_table = list(A)
group_table = list(B)

atomic_num_Al, group_num_Al, period_num_Al = element_repre (symbol_table, group_table, 'Al')
