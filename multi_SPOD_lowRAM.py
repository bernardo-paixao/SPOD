import os
os.environ["OMP_NUM_THREADS"] = "24" # export OMP_NUM_THREADS=20
os.environ["OPENBLAS_NUM_THREADS"] = "24" # export OPENBLAS_NUM_THREADS=20
os.environ["MKL_NUM_THREADS"] = "24" # export MKL_NUM_THREADS=20
os.environ["VECLIB_MAXIMUM_THREADS"] = "24" # export VECLIB_MAXIMUM_THREADS=20
os.environ["NUMEXPR_NUM_THREADS"] = "24" # export NUMEXPR_NUM_THREADS=20

import numpy as np
import lvpyio as lv
from scipy.fft import fft, fftfreq
from scipy.linalg import eigh
import pickle

def get_coordinates(set):
    buffer = set[0]
    # print(buffer.attributes["AcqTimeSeries"])
    frame = buffer[0]
    coordinates, _, _ = calculate_coordinates(frame)
    # Get x, y, z and velocities
    X = np.array(coordinates[0])
    Y = np.array(coordinates[1])
    return X, Y

def get_velocity(Set,Nx,Ny,buffer_id,xlims,ylims):
    Vx = np.zeros((len(buffer_id),Ny*Nx))
    Vy = np.zeros((len(buffer_id),Ny*Nx))
    Vz = np.zeros((len(buffer_id),Ny*Nx))
    for j, i in enumerate(buffer_id):
        i = int(i)
        Buffer = Set[i]
        frame = Buffer[0]
        scaled_masked_array = frame.as_masked_array()
        displacements = calculate_displacements(scaled_masked_array)
        Vx[j,:] = crop_field(np.array(displacements[0]),xlims,ylims).flatten()
        Vy[j,:] = crop_field(np.array(displacements[1]),xlims,ylims).flatten()
        Vz[j,:] = crop_field(np.array(displacements[2]),xlims,ylims).flatten()
    if len(buffer_id)==1:
        Vx = np.squeeze(Vx,axis=0)
        Vy = np.squeeze(Vy,axis=0)
        Vz = np.squeeze(Vz,axis=0)
    return Vx, Vy, Vz

def calculate_displacements(array):
    height, width = array.shape
    is_3c = len(array.dtype) == 3

    vector_components = ["u", "v", "w"] if is_3c else ["u", "v"]

    displacements = np.empty((3 if is_3c else 2, height, width), dtype=np.float64)
    for i, key in enumerate(vector_components):
        displacements[i] = array[key]

    return displacements

def calculate_coordinates(frame):
    # Height is the y-dimension.
    # Not to be confused with the Height scalar field stored in TS:Height.
    height, width = frame.shape

    if frame.is_3c:
        scales = [frame.scales.x, frame.scales.y, frame.scales.z]
    else:
        scales = [frame.scales.x, frame.scales.y]

    coordinates = np.empty((3 if frame.is_3c else 2, height, width), dtype=np.float64)

    coordinates_x = frame.grid.x * np.arange(width) + frame.grid.x // 2
    for i in range(height):
        coordinates[0, i, :] = coordinates_x
    coordinates[0] = scales[0].slope * coordinates[0] + scales[0].offset

    coordinates_y = frame.grid.y * np.arange(height) + frame.grid.y // 2
    for i in range(width):
        coordinates[1, :, i] = coordinates_y
    coordinates[1] = scales[1].slope * coordinates[1] + scales[1].offset

    if frame.is_3c:
        coordinates[2] = scales[2].slope *1 + scales[2].offset # Replace 1 for frame[0]["TS:Height"]

    units = [scale.unit for scale in scales]
    descriptions = [scale.description for scale in scales]
    return coordinates, units, descriptions

def get_coord_limits(X,Y,limx=(-0.6,0.6),limy=(-0.6,0.6)):
    xinf = np.argmin((X[0,:]-limx[0])**2)
    xsup = np.argmin((X[0,:]-limx[-1])**2)
    ysup = np.argmin((Y[:,0]-limy[0])**2)
    yinf = np.argmin((Y[:,0]-limy[1])**2)
    return [xinf,xsup], [yinf,ysup]

def crop_field(F,xlims,ylims):
    return F[ylims[0]:ylims[1],xlims[0]:xlims[1]]

def build_Qhat(case_name, file_path, mean_path, Nfft, overlap, D, U):
    dest_path = '/mnt/rozo/2atgobain/SPOD data/3rd_campaign/'+case_name+'_Nfft'+str(Nfft) 
    if  os.path.isdir('/mnt/rozo/2atgobain/SPOD data/Qhat'):
        dir = os.listdir('/mnt/rozo/2atgobain/SPOD data/Qhat')  
        # if len(dir) != 0: 
        #     raise ValueError('Qhat directory is not empty.')
    else:
        os.mkdir('./Qhat')
    
    input_set = lv.read_set(file_path)
    avg_set = lv.read_set(mean_path)
    X, Y = get_coordinates(input_set)
    #print(f'Nx = {np.shape(X)[1]}, Ny = {np.shape(X)[0]}')
    X *= 1/(D*1e3)
    Y *= 1/(D*1e3)
    xlims, ylims = get_coord_limits(X,Y)
    X = crop_field(X,xlims,ylims)
    Y = crop_field(Y,xlims,ylims)

    with open(dest_path+'/'+case_name+'_X.txt','wb') as f:
            pickle.dump(X,f)
    with open(dest_path+'/'+case_name+'_Y.txt','wb') as f:
            pickle.dump(Y,f)

    Ny, Nx = np.shape(X)
    #print(f'Nx cropped = {Nx}, Ny cropped = {Ny}')
    avg_Vx, avg_Vy, avg_Vz = get_velocity(avg_set,Nx,Ny,[0],xlims,ylims)
    #print(f'Shape avg_V = {np.shape(avg_Vx)}')
    
    Ns = len(input_set) # Number of samples
    if Ns<50000:
        raise ValueError('Not enough velocity fields')
    Nblk = int((Ns-Nfft*overlap)//(Nfft*(1-overlap)))
    #print(f'Number of blocks = {Nblk}')
    
    buffer = input_set[0]
    frame = buffer[0]
    t_str = frame.attributes['AcqTimeSeries']
    t0 = float(t_str[0:-3])*1e-6
    buffer = input_set[1]
    frame = buffer[0]
    t_str = frame.attributes['AcqTimeSeries']
    t1 = float(t_str[0:-3])*1e-6
    dt = t1-t0
    f_vect = fftfreq(Nfft,d=dt)

    with open(dest_path+'/'+case_name+'_Freq.txt','wb') as f:
        pickle.dump(f_vect,f)
    
    window = np.hanning(Nfft) # window function (Welch's method)
    Cw = np.sqrt(Nfft/np.sum(window**2))
    
    Q_hat = np.zeros((Nfft,3*Nx*Ny,Nblk),dtype='complex')

    for i in range(Nblk):
       id_list = np.arange(Nfft)+Nfft*(1-overlap)*i
       Vx, Vy, Vz = get_velocity(input_set,Nx,Ny,id_list,xlims,ylims)
       Vx = Vx-avg_Vx # -np.mean(Vx,axis=0) # substract block mean
       Vy = Vy-avg_Vy # -np.mean(Vy,axis=0)
       Vz = Vz-avg_Vz # -np.mean(Vz,axis=0)
       q = np.concatenate((Vx,Vy,Vz),axis=1)
       Q_hat[:,:,i] = fft(q*window[:,np.newaxis]*Cw,Nfft,axis=0)
       #print(f'Building Q_hat, {i}/{Nblk}')
       #print(f'Block average, Vx : {np.mean(Vx).round(3)}, Vy : {np.mean(Vy).round(3)}, Vz : {np.mean(Vz).round(3)}')
   
    for i in range(Nfft):
        with open('/mnt/rozo/2atgobain/SPOD data/Qhat/'+case_name+'_Qhat_'+str(i)+'.txt','wb') as f:
            pickle.dump(Q_hat[i,:,:],f)
    
    
def load_Qhat(case_name, id_Nfft):
    with open('/mnt/rozo/2atgobain/SPOD data/Qhat/'+case_name+'_Qhat_'+str(id_Nfft)+'.txt','rb') as f:
        qhat_i = pickle.load(f)    
    return qhat_i

def get_mesh_specs(file_path,D):
    input_set = lv.read_set(file_path)
    X, Y = get_coordinates(input_set)
    X *= 1/(D*1e3)
    Y *= 1/(D*1e3)
    xlims, ylims = get_coord_limits(X,Y)
    X = crop_field(X,xlims,ylims)
    Y = crop_field(Y,xlims,ylims)
    dx = X[0,1]-X[0,0]
    dy = Y[1,0]-Y[0,0]
    Ny, Nx = np.shape(X)
    return dx, dy, Nx, Ny

def main(create_qhat = True):

    D = 62.5*1e-3 # Jet diameter [m]
    U = 29  # Jet reference axial speed [m/s], 29 if Ar=0.3 and 69 if Ar=0.1
    Nfft = 1024 
    overlap = 0.5 # overlap: 0.5 -> 50%

    case_list = ['Ar03_S0_1000Pa_d5mm_12p5kHz']
    
    Qhat_folder =  '/mnt/rozo/2atgobain/SPOD data/Qhat'   
    
    for j, case_name in enumerate(case_list):
        print("Processing : " + case_name)
        folder_path = '/mnt/rozo/2atgobain/Jet100_2D3C/Essai_3/JET100_Gobain_Fast_SPIV_CameraAjust/'+case_name+'/'
        file_name = 'StereoPIV_MPd(2x16x16_50%ov).set'
        mean_path = folder_path + 'StereoPIV_MPd(2x16x16_50%ov)/Avg_StdDev.set'
        file_path = folder_path+file_name
        dest_path = '/mnt/rozo/2atgobain/SPOD data/3rd_campaign/'+case_name+'_Nfft'+str(Nfft)  

        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
    
        if create_qhat:
            build_Qhat(case_name,file_path,mean_path, Nfft, overlap, D,U)
        
        #Q_hat = load_Qhat(case_name,0)
        dx,dy,Nx,Ny = get_mesh_specs(file_path, D)

        W = np.diag(np.ones(3*Nx*Ny))*dx*dy*D**2

        # Since the time data was not complex, the spectrum is duplicated, we only need Nfft/2 
        for i in range(Nfft//2):
            Q_hat = load_Qhat(case_name,i)
            Q_hat_herm = Q_hat.T.conj()
            M = Q_hat_herm@W@Q_hat
            Lambda, Psi = eigh(M)
            Phi = Q_hat@Psi
            idx_sort = np.argsort(-Lambda)

            #print(f'Computing SPOD modes, {i}/{Nfft//2}')

            with open(dest_path+'/'+case_name+'_Lambda_'+str(i)+'.txt','wb') as f:
                pickle.dump(Lambda[idx_sort],f)
            with open(dest_path+'/'+case_name+'_Phi_'+str(i)+'.txt','wb') as f:
                pickle.dump(Phi[:,idx_sort],f)
        
        for filename in os.listdir(Qhat_folder):
            qhat_path = os.path.join(Qhat_folder, filename)
            try:
                if os.path.isfile(qhat_path) or os.path.islink(qhat_path):
                    os.unlink(qhat_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (qhat_path, e))

if __name__=="__main__":
    main()


      
    
    
