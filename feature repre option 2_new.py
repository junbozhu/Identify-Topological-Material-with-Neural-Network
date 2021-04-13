import numpy as np

def element_repre(symbol_table, group_table, symbol_input_str): # input like 'Na'
    # input element symbol, output its atomic number, group number, period number
    if symbol_input_str in symbol_table:
        index = symbol_table.index(symbol_input_str)
        atomic_num = index + 1
        group_num = int(group_table[index])
        if group_num == 101:
            group_num = 19
        elif group_num == 102:
            group_num =20
    else: print('error, element not in periodic table')
    # note: in this way we regard all rare earth as period=6, group=19. But we don't cover magnetic material so maybe ok
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

def lattice_group(spacegroup_num):
    if spacegroup_num <=2:
        return 1 #triclinic
    elif 2< spacegroup_num <=15:
        return 2 #Monoclinic
    elif 15< spacegroup_num <=74:
        return 3 #Orthorhombic
    elif 74< spacegroup_num <=142:
        return 4 #Tetragonal
    elif 142< spacegroup_num <=167:
        return 5 #Trigonal
    elif 167< spacegroup_num <=194:
        return 6 #Hexagonal
    elif 194< spacegroup_num <=230:
        return 7 #Cubic
    else:
        print('Spacenumber error')
        return 0
    
# load neccesary table for element_repre function 
A,B = np.loadtxt("ref_periodic_table.txt", unpack = True,dtype=str)
symbol_table = list(A) # 109 elements in total
group_table = list(B)

## load labels:

x, y = np.loadtxt("labels.txt", unpack=True)
labels_binary_soc = np.array([x])
labels_binary_nsoc = np.array([y])

A,B = np.loadtxt("classes.txt", unpack = True,dtype=str)
class_soc = np.array([A])
class_nsoc = np.array([B])

C = np.loadtxt("idstr.txt", unpack = True,dtype=str)
idstr = np.array([C])

N = len(x)

## load features:
spacegroup = np.array([np.loadtxt("data_spacegroup.txt", unpack=True)])
icsd_id_str = list(np.loadtxt("icsd_id_str.txt",delimiter='/n',dtype=str))
mp_id_str = list(np.loadtxt("mp_id_str.txt",delimiter='/n',dtype=str))
formula = list(np.loadtxt("formula.txt",delimiter='/n',dtype=str))
elements = list(np.loadtxt("elements.txt",delimiter='/n',dtype=str))
elements_num = list(np.loadtxt("elements_num.txt",delimiter='/n',dtype=str))
nelec = np.array([np.loadtxt("nelec.txt", unpack=True)])
nsites = np.array([np.loadtxt("nsites.txt", unpack=True)])

'''feature representation:
spacegroup: one hot of lattice type
elements:
    one hot of group (18+2) and period (7)
    one hot of atomic number (109)
    note: actually weighted one-hot: Bi2Se3 will be 0.4 Bi and 0.6 Se 
nelec: float, raw to standard (x - average) / sigma
nsites: float, same as nelec
'''

# representations
feature_atomic_num = np.zeros([109,N]) # 109 elements, weighted one hot
feature_atomic_group = np.zeros([20,N]) # 20 groups, weighted one hot
feature_atomic_period = np.zeros([7, N]) # 7 periods, weighted one hot
feature_lattice = np.zeros([7,N]) # 7 spacegroup types, weighted one hot
feature_nelec = (nelec - np.average(nelec))/np.std(nelec) # number of electrons in unit cell, standardize
feature_nsites = (nsites - np.average(nsites))/np.std(nsites) # number of atoms in unit cell, standardize

# construct representations
for i in range(N):
    elem_list = elements[i].lstrip('[\'').rstrip('\']').split('\', \'') # split the string into list of element strings
    n_elem = len(elem_list) # number of elements in the formula
    elem_num_list = elements_num[i].lstrip('[\'').rstrip('\']').split(', ') # split the input string
    elem_num_list = [int(k) for k in elem_num_list] # convert this list of string into list of int
    #print(elem_list) # for testing
    #print(elem_num_list) # for testing
    for j in range(n_elem): # for each of the element in the formula
        atomic_num, group_num, period_num = element_repre(symbol_table, group_table, elem_list[j])
        weight = elem_num_list[j]/sum(elem_num_list) #weight ratio of this element
        #print(weight) # for testing
        feature_atomic_num[atomic_num-1,i] = weight
        feature_atomic_group[group_num-1,i] += weight # there can be multiple elements in same group so add it
        feature_atomic_period[period_num-1,i] += weight # there can be multiple elements in same period so add it
    feature_lattice[ lattice_group(spacegroup[0,i])-1 , i] = 1 #one hot
    if i%1000 == 1:
        print(str(i-1)+' transformed')

print("Transformation finished")


# construct the complete data representation
# simply add the dimensions up
# totoal dimension = 109+20+7+7+1+1 = 145

data = np.zeros([145,N])
data[0:109,:] = feature_atomic_num
data[109:129,:] = feature_atomic_group
data[129:136,:] = feature_atomic_period
data[136:143,:] = feature_lattice
data[143:144,:] = feature_nelec
data[144:145,:] = feature_nsites

with open('data_repre2.txt','w') as f: # note: will overwrite
        np.savetxt(f, data.T,fmt='%.5f',header="data_repre_option_2_new_2021")

