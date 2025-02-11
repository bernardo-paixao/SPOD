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
from scipy.interpolate import griddata
import pickle
import matplotlib.pyplot as plt

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
    dest_path = '/mnt/rozo/2atgobain/SPOD data/'+case_name+'_Nfft'+str(Nfft)+'_fourier'
    if  os.path.isdir('/mnt/rozo/2atgobain/SPOD data/Qhat'):
        dir = os.listdir('/mnt/rozo/2atgobain/SPOD data/Qhat')  
        #if len(dir) != 0: 
        #    raise ValueError('Qhat directory is not empty.')
    else:
        os.mkdir('/mnt/rozo/2atgobain/SPOD data/Qhat')
    
    input_set = lv.read_set(file_path)
    avg_set = lv.read_set(mean_path)
    X, Y = get_coordinates(input_set)
    print(f'Nx = {np.shape(X)[1]}, Ny = {np.shape(X)[0]}')
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
    print(f'Nx cropped = {Nx}, Ny cropped = {Ny}')
    avg_Vx, avg_Vy, avg_Vz = get_velocity(avg_set,Nx,Ny,[0],xlims,ylims)
    print(f'Shape avg_V = {np.shape(avg_Vx)}')
    
    Ns = len(input_set) # Number of samples
    if Ns<50000:
        raise ValueError('Not enough velocity fields')
    Nblk = int((Ns-Nfft*overlap)//(Nfft*(1-overlap)))
    print(f'Number of blocks = {Nblk}')
    
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
    Cw = np.sqrt(Nfft/np.sum(window**2)) # Energy correction factor
    
    Q_hat = np.zeros((Nfft,3*Nx*Ny,Nblk),dtype='complex')

    for i in range(Nblk):
       id_list = np.arange(Nfft)+Nfft*(1-overlap)*i
       Vx, Vy, Vz = get_velocity(input_set,Nx,Ny,id_list,xlims,ylims)
       Vx = Vx-avg_Vx # -np.mean(Vx,axis=0) # substract block mean
       Vy = Vy-avg_Vy # -np.mean(Vy,axis=0)
       Vz = Vz-avg_Vz # -np.mean(Vz,axis=0)
       q = np.concatenate((Vx,Vy,Vz),axis=1)
       Q_hat[:,:,i] = fft(q*window[:,np.newaxis]*Cw,Nfft,axis=0)
       print(f'Building Q_hat, {i}/{Nblk}')
       print(f'Block average, Vx : {np.mean(Vx).round(3)}, Vy : {np.mean(Vy).round(3)}, Vz : {np.mean(Vz).round(3)}')
   
    for i in range(Nfft):
        with open('/mnt/rozo/2atgobain/SPOD data/Qhat/'+case_name+'_Qhat_'+str(i)+'.txt','wb') as f:
            pickle.dump(Q_hat[i,:,:],f)
    
    
def load_Qhat(case_name, id_Nfft):
    with open('/mnt/rozo/2atgobain/SPOD data/Qhat/'+case_name+'_Qhat_'+str(id_Nfft)+'.txt','rb') as f:
        qhat_i = pickle.load(f)    
    return qhat_i

def load_freq(dest_path,case_name):
    with open(dest_path+'/'+case_name+'_Freq.txt','rb') as f:
        freq = pickle.load(f)
    return freq

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
    return dx, dy, Nx, Ny, X, Y

def main(create_qhat = True):

    D = 62.5*1e-3 # Jet diameter [m]
    U = 69  # Jet reference axial speed [m/s], 29 if Ar=0.3 and 69 if Ar=0.1
    Nfft = 1024 
    overlap = 0.5 # overlap: 0.5 -> 50%
    Ntheta = 64

    Nsave = 3

    Qhat_folder =  '/mnt/rozo/2atgobain/SPOD data/Qhat'
    case_list = ['Ar01_S0_4500Pa_12p5kHz_d5mm',
                 'Ar01_S0_4500Pa_12p5kHz_d11mm',
                 'Ar01_S0_4500Pa_12p5kHz_d21mm',
                 'Ar01_S0_4500Pa_12p5kHz_d31mm',
                 'Ar01_S0_4500Pa_12p5kHz_d41mm',
                 'Ar01_S0_4500Pa_12p5kHz_d51mm',
                 'Ar01_S0_4500Pa_12p5kHz_d61mm',
                 'Ar01_S0_4500Pa_12p5kHz_d81mm',
                 'Ar01_S0_4500Pa_12p5kHz_d101mm',
                 'Ar01_S0_4500Pa_12p5kHz_d121mm',
                 'Ar01_S0_4500Pa_12p5kHz_d201mm']
    
    for k, case_name in enumerate(case_list):
    
        folder_path = '/mnt/rozo/2atgobain/Jet100_2D3C/Essai_4/SPIV_SaintGobain/'+case_name+'/'
        file_name = 'StereoPIV_MPd(2x16x16_50%ov).set'
        mean_path = folder_path + 'StereoPIV_MPd(2x16x16_50%ov)/Avg_StdDev.set'
        file_path = folder_path+file_name
        dest_path = '/mnt/rozo/2atgobain/SPOD data/'+case_name+'_Nfft'+str(Nfft)+'_fourier'  

        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
    
        if create_qhat:
            # Build Qhat matrix 
            build_Qhat(case_name,file_path,mean_path, Nfft, overlap, D,U)
            print("Done building Qhat matrix")

        # Load Qhat 0 freq from disk
        Q_hat = load_Qhat(case_name,0)
        _, Nblk = np.shape(Q_hat)
        freq = load_freq(dest_path,case_name)
        dx,dy,Nx,Ny, X, Y = get_mesh_specs(file_path, D)
        print(f"Nx = {Nx}, Ny = {Ny}")
        print(f'Shape of Qhat = {np.shape(Q_hat)}')

        Tcart = np.arctan2(Y,X)

        # Create polar mesh
        Nr = Nx//2
        print(f'Nr={Nr}, Ntheta={Ntheta}')

        theta1d = np.linspace(0,2*np.pi,Ntheta)
        r1d = np.linspace(0,np.max(X, axis=None)-0.01,Nr)
        dr = r1d[1]-r1d[0]
        RR , TTheta = np.meshgrid(r1d,theta1d)
        m = np.fft.fftfreq(Ntheta,1/Ntheta)
        
        W_polar = np.diag(np.tile(2*r1d*dr+dr**2,3))*np.pi
        print(f'Shape W_polar : {np.shape(W_polar)}')

        # Since the time data was not complex, the spectrum is duplicated, we only need Nfft/2 
        Q_hat_polar = np.zeros((Ntheta,3*Nr,Nblk),dtype='complex')
        Lam_mat = np.zeros((Nfft//2,Ntheta,Nblk),dtype='float')
        Phi_mat = np.zeros((Nfft//2,Ntheta,3*Nr,Nsave),dtype='complex')
        for i in range(Nfft//2):
            if i ==0:
                Q_hat = load_Qhat(case_name,i)
            else:
                Q_hat = 2*load_Qhat(case_name,i)
            
            for j in range(Nblk):
                Qr_cart = Q_hat[0:Ny*Nx,j].reshape(Ny,Nx)*np.cos(Tcart) +  Q_hat[Ny*Nx:2*Ny*Nx,j].reshape(Ny,Nx)*np.sin(Tcart)
                Qtheta_cart = -Q_hat[0:Ny*Nx,j].reshape(Ny,Nx)*np.sin(Tcart) +  Q_hat[Ny*Nx:2*Ny*Nx,j].reshape(Ny,Nx)*np.cos(Tcart)
                
                Qr = griddata((X.flatten(),Y.flatten()), Qr_cart.flatten(), (RR*np.cos(TTheta), RR*np.sin(TTheta)), method='linear').reshape(Ntheta,Nr)
                Qtheta = griddata((X.flatten(),Y.flatten()), Qtheta_cart.flatten(), (RR*np.cos(TTheta), RR*np.sin(TTheta)), method='linear').reshape(Ntheta,Nr)
                Qz = griddata((X.flatten(),Y.flatten()), Q_hat[2*Ny*Nx::,j].flatten(), (RR*np.cos(TTheta), RR*np.sin(TTheta)), method='linear').reshape(Ntheta,Nr)
                
                Qr_fft = fft(Qr,axis=0)
                Qtheta_fft = fft(Qtheta,axis=0)
                Qz_fft = fft(Qz,axis=0)

                Q_hat_polar[:,:,j] = np.concatenate((Qr_fft,Qtheta_fft,Qz_fft),axis=1)
            
            print(f'Computing SPOD modes, {i}/{Nfft//2}')
            
            for j in range(Ntheta):
                # Snapschot method
                Q_hat_herm = Q_hat_polar[j,:,:].T.conj()
                M = Q_hat_herm@W_polar@Q_hat_polar[j,:,:]
                Lambda, Psi = eigh(M)

                # Recovering the actual SPOD modes
                Phi = Q_hat_polar[j,:,:]@Psi

                # Sorting highest amplitude modes
                idx_sort = np.argsort(-Lambda)
                Phi = Phi[:,idx_sort]
                Lambda = Lambda[idx_sort]

                # Storing the results in big matrices
                Phi_mat[i,j,:,:] = Phi[:,0:Nsave]
                Lam_mat[i,j,:] = Lambda
        
        # Store results in a dictionary
        results = {'lambda': Lam_mat,
                    'phi':Phi_mat,
                    'm': m,
                    'freq': freq,
                    'r': r1d}
        
        # Save results
        with open(dest_path+'/'+case_name+'_SPOD_results.txt','wb') as f:
            pickle.dump(results,f)

        for filename in os.listdir(Qhat_folder):
            qhat_path = os.path.join(Qhat_folder, filename)
            try:
                if os.path.isfile(qhat_path) or os.path.islink(qhat_path):
                    os.unlink(qhat_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (qhat_path, e))

if __name__=="__main__":
    main()


      
    
    
