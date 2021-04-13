import requests
import numpy as np
import pprint

# get all the data available on Materiae

token = ('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE1NzI0OTEyMjMsIm5iZiI6MTU3MjQ5MTIyMywianRpIjoiODQ5NWJjNzYtMDQzYy00Mjk3LWJiNTItZmY4NzhhMTc3NjI4IiwiZXhwIjoxNTcyNTc3NjIzLCJpZGVudGl0eSI6InpqYmRkeEBnbWFpbC5jb20iLCJmcmVzaCI6ZmFsc2UsInR5cGUiOiJhY2Nlc3MifQ.cT1-AvqvScr-PSQCqIrtna1MYS2t0YlrjD_-hOe6F04')
         
hostname = 'materiae.iphy.ac.cn'
headers = {'Authorization': 'Bearer %s' % token}

## load data:

x, y = np.loadtxt("labels.txt", unpack=True)
labels_binary_soc = np.array([x])
labels_binary_nsoc = np.array([y])

A,B = np.loadtxt("classes.txt", unpack = True,dtype=str)
class_soc = np.array([A])
class_nsoc = np.array([B])

C = np.loadtxt("idstr.txt", unpack = True,dtype=str)
idstr = np.array([C])

N = len(x)

'''
# Data options:
spacegroup = np.zeros([1,N])
icsd_id_str = [] # e.g. [67559] 
mp_id_str = [] # eg. mp-6999
formula = [] # store as a list of string
elements = [] # stored as e.g. ['P', 'S', 'Sc']
elements_num = [] # stored as e.g. ['2', '8', '2']
nelec = np.zeros([1,N]) # probably not useful??
nsites = np.zeros([1,N]) # probably not useful??
## detailed structure info are not on materiae!
'''

# continuing:
spacegroup = np.array([np.loadtxt("data_spacegroup.txt", unpack=True)])
icsd_id_str = list(np.loadtxt("icsd_id_str.txt",delimiter='/n',dtype=str))
mp_id_str = list(np.loadtxt("mp_id_str.txt",delimiter='/n',dtype=str))
formula = list(np.loadtxt("formula.txt",delimiter='/n',dtype=str))
elements = list(np.loadtxt("elements.txt",delimiter='/n',dtype=str))
elements_num = list(np.loadtxt("elements_num.txt",delimiter='/n',dtype=str))
nelec = np.array([np.loadtxt("nelec.txt", unpack=True)])
nsites = np.array([np.loadtxt("nsites.txt", unpack=True)])


# get detailed info
start = len(formula)
i = start
print('start at '+ str(start))
while i < N:
    url = "http://%s/api/materials/%s" % (hostname, idstr[0,i])
    #print(url)
    params = {'fields': 'spacegroup.number,icsd_ids,mp_id,formula,elements,elements_num,nelec,nsites'}
    headers = {'Authorization': 'Bearer %s' % token}
    response = requests.get(url=url, params=params, headers=headers)
    #pprint.pprint(response.json())
    # take all the info
    spacegroup[0,i] = response.json()['spacegroup.number']
    icsd_id_str = icsd_id_str + [response.json()[ 'icsd_ids' ]]
    mp_id_str = mp_id_str + [response.json()[ 'mp_id' ]]
    formula = formula + [ response.json()[ 'formula' ]]
    elements = elements + [ response.json()[ 'elements' ] ]
    elements_num = elements_num + [ response.json()[ 'elements_num' ] ]
    nelec[0,i] = response.json()[ 'nelec' ]
    nsites[0,i] = response.json()[ 'nsites' ]
    if i%50 == 49:
        print(str(i+1) + ' enries finished.')
    i = i+1
    
# save all those info

for j in range(1):
    with open('data_spacegroup.txt','w') as f: # note: will overwrite
        np.savetxt(f, spacegroup.T,fmt='%.1f',header="spacegroup")
    
    with open('icsd_id_str.txt','w') as f: # note: will overwrite
        np.savetxt(f, icsd_id_str, fmt='%s', header="icsd_id_str")

    with open('mp_id_str.txt','w') as f: # note: will overwrite
        np.savetxt(f, mp_id_str, fmt='%s', header="mp_id_str")    

    with open('formula.txt','w') as f: # note: will overwrite
        np.savetxt(f, formula, fmt='%s', header="formula")    

    with open('elements.txt','w') as f: # note: will overwrite
        np.savetxt(f, elements, fmt='%s', header="elements")

    with open('elements_num.txt','w') as f: # note: will overwrite
        np.savetxt(f, elements_num, fmt='%s', header="elements_num")

    with open('nelec.txt','w') as f: # note: will overwrite
        np.savetxt(f, nelec.T, fmt='%s', header="nelec")

    with open('nsites.txt','w') as f: # note: will overwrite
        np.savetxt(f, nsites.T, fmt='%s', header="nsites")

