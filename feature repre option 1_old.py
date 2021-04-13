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
symbol_table = list(A)
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
elements: one hot of group and period, if multiple elements then many 1's
nelec: float, raw to standard (x - average) / sigma
nsites: float, same as nelec
'''

'''
# test for 1 entry
i = 10
# convert str text "['N', 'O', 'Sr', 'W']"  into list of string ['N', 'O', 'Sr', 'W']
elem_list = elements[i].lstrip('[\'').rstrip('\']').split('\', \'')
feature_elem = np.zeros([7,20]) # the position at preiodic table (period, group), 7 periods 18+2 group
for elem in elem_list:
    atomic_num, group_num, period_num = element_repre(symbol_table, group_table, elem)
    feature_elem[period_num-1,group_num-1] = 1
feature_elem = np.reshape( feature_elem, [140,1], order='F' )
        # 'F' means  the first index changing fastest, and the last index changing slowest
feature_lattice = np.zeros([7,1])
feature_lattice[ lattice_group(spacegroup[0,i])-1 , 0] = 1 #one hot
# test passed
'''

# transform all data
feature_elem = np.zeros([7,20,N])
feature_lattice = np.zeros([7,N])
feature_nelec = (nelec - np.average(nelec))/np.std(nelec)
feature_nsites = (nsites - np.average(nsites))/np.std(nsites)

for i in range(N):
    elem_list = elements[i].lstrip('[\'').rstrip('\']').split('\', \'')
    for elem in elem_list:
        atomic_num, group_num, period_num = element_repre(symbol_table, group_table, elem)
        feature_elem[period_num-1,group_num-1,i] = 1
    feature_lattice[ lattice_group(spacegroup[0,i])-1 , i] = 1 #one hot

feature_elem_3D = feature_elem
feature_elem = np.reshape(feature_elem_3D,[140,N],order='F')

# make up data
# feature_elem: 140xN
# feature_lattice: 7xN
# feature_nelec: 1xN
# feature_nsites: 1xN


# easy way: not factorize, simply add the dimensions up
# totoal dimension 149
data1 = np.zeros([149,N])
data1[0,:] = feature_nelec
data1[1,:] = feature_nsites
data1[2:9,:] = feature_lattice
data1[9:149,:] = feature_elem

with open('data_repre1.txt','w') as f: # note: will overwrite
        np.savetxt(f, data1.T,fmt='%.1f',header="data_repre_option_1")


# second feature repre:
# dimension = 140*7 + 2 
