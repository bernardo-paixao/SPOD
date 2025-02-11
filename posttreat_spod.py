import numpy as np
import pickle
from scipy.signal import find_peaks

case_prefix = 'Ar03_S0_1000Pa_d'
case_sufix = 'mm_12p5kHz'
pos_list = [31]

for pos in pos_list:
    case_name = case_prefix + str(pos) + case_sufix
    print('Processing : ' + case_name)
    folder_path ='/mnt/rozo/2atgobain/SPOD data/Nfft8192/'+case_name+'_Nfft8192/' 
    
    #U = 29
    D = 62*1e-3
    
    with open(folder_path+case_name+'_Freq.txt','rb') as f:
        freq = pickle.load(f)
    with open(folder_path+case_name+'_X.txt','rb') as f:
        X = pickle.load(f)
    with open(folder_path+case_name+'_Y.txt','rb') as f:
        Y = pickle.load(f)
    with open(folder_path+case_name+'_Nfft8192_Lambda_0.txt','rb') as f:
            lam0 = pickle.load(f)

    Nblk = len(lam0)
    Nfft = len(freq)//2
    lam = np.zeros((Nfft,Nblk),dtype='complex')
    for i in range(Nfft): #Nfft
        with open(folder_path+case_name+'_Nfft8192_Lambda_'+str(i)+'.txt','rb') as f:
            lam[i,:] = pickle.load(f)
    Ek = np.sum(np.real(lam),axis=None)

    lead_modes = np.real(lam[:,-1])/Ek
    peaks, _ = find_peaks(lead_modes,height=5e-4,distance=4)

    #print(f'Peaks at St = {np.round(freq[peaks]*D/U,3)}')
    #print(f'dSt = {freq[1]*D/U}')

    Ny, Nx = np.shape(X) 
    Phi_list = []
    for i in range(Nfft): 
        with open(folder_path+case_name+'_Nfft8192_Phi_'+str(i)+'.txt','rb') as f:
            phi = pickle.load(f)
            
        Phi_list.append(phi[:,Nblk-2::])

    results = {'lambda': lam,
            'phi': Phi_list,
            'freq': freq,
            'X': X,
            'Y': Y,
            'peaks' : peaks}

    with open('/mnt/rozo/2atgobain/SPOD data/Nfft8192/'+case_name+'_N8192_SPODmodes.txt','wb') as f:
        pickle.dump(results,f)


