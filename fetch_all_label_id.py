import requests
import numpy as np

token = ('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1NzE1ODU4MTksIm5iZiI6MTU3MTU4NTgxOSwianRpIjoiNjc0NWQ2MTItMmVjZC00NzM2LTg2YWYtMWY5YzMyYzZiZTI0IiwiZXhwIjoxNTcxNjcyMjE5LCJpZGVudGl0eSI6InpqYmRkeEBnbWFpbC5jb20iLCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.AR3fu-SW4PsjJ8dMaulcD4Uy6UKXwzfB4GEjBTC8Elo')
         
hostname = 'materiae.iphy.ac.cn'

url = "http://%s/api/materials" % hostname

## Fetch All Material (with only default important fields)
params = {}

headers = {'Authorization': 'Bearer %s' % token}
response = requests.get(url=url, params=params, headers=headers)

count = response.json()['count']
print(count)
mats = response.json()['materials']
#mats contain 28605 materials, mats is a list of dict, one dict for one compound
#print(mats[0])
# {'id': 'MAT00000001',
#'formula': 'Na5P3(H3O4)4',
#'spacegroup.number': 2,
#'nsoc_topo_class': 'Triv_Ins',
#'soc_topo_class': 'Triv_Ins'}

N = count # total number of my data, it's just 28605 =  len(mats)

# now set labels:
class_soc = []
class_nsoc = []
labels_binary_soc = np.zeros([1,N])
labels_binary_nsoc = np.zeros([1,N])
idstr = []


for i in range(N):
    #print('No.' + str(i))
    idstr = idstr + [mats[i]['id']]
    
    if mats[i]['soc_topo_class'] == '':
        class_soc = class_soc + ['Unknown']
    else:   class_soc = class_soc + [mats[i]['soc_topo_class']]
    #print('soc_topo_class: '+ class_soc[i])

    if mats[i]['nsoc_topo_class'] == '':
        class_nsoc = class_nsoc + ['Unknown']
    else:   class_nsoc = class_nsoc + [mats[i]['nsoc_topo_class']]
    #print('nsoc_topo_class: '+class_nsoc[i])
    
    if mats[i]['soc_topo_class'] != 'Triv_Ins' and mats[i]['soc_topo_class'] != '':
        labels_binary_soc[0,i] = 1 # topological so label change to 1
    if mats[i]['nsoc_topo_class'] != 'Triv_Ins'and mats[i]['nsoc_topo_class'] != '':
        labels_binary_nsoc[0,i] = 1 # topological so label change to 1
        
    #print('soc_topo_label: '+ str(labels_binary_soc[0,i]))
    #print('nsoc_topo_label: '+str(labels_binary_nsoc[0,i]))

with open('labels.txt','w') as f: # note: will overwrite
    data = np.array([labels_binary_soc[0,:], labels_binary_nsoc[0,:]]).T
    np.savetxt(f, data,
               fmt='%.1f',
               header="labels_binary_soc labels_binary_nsoc")
with open('classes.txt','w') as f: # note: will overwrite
    data = np.column_stack((class_soc, class_nsoc))
    np.savetxt(f,data,fmt='%s',header="class_soc class_nsoc")

with open('idstr.txt','w') as f: # note: will overwrite
    np.savetxt(f,idstr,fmt='%s',header="idstr")

## load method:
##x, y = np.loadtxt("labels.txt", unpack=True)
##A,B = np.loadtxt("classes.txt", unpack = True,dtype=str)
## C = np.loadtxt("idstr.txt", unpack = True,dtype=str)


