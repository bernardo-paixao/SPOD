import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as clrs
from scipy import stats
import pickle


def main():

    plot_idv = False
    plt.rcParams.update({'font.size': 18})

    D = 62.5*1e-3 # Jet diameter [m]
    U = 69  # Jet reference axial speed [m/s], 29 if Ar=0.3 and 69 if Ar=0.1
    
    case_root = 'Ar01_S04_4500Pa_12p5kHz'
    d_list = np.array([5,11,21,31,41,51,61,81,121,201])
    plot_m = [2,1,0,-1,-2]
    id_follow = np.zeros((len(d_list),len(plot_m)),dtype='int')
    freq_selection = [60,6,6,21,18]
 
    for k, d in enumerate(d_list):
        case_name = case_root+'_d'+ str(d)+'mm'
        folder_path = '/mnt/rozo/2atgobain/SPOD data/'+case_name+'_Nfft1024_fourier'
        fig_name = case_name[0:8]+case_name[23::]+'_n64'
        print(fig_name)

        with open(folder_path+'/'+case_name+'_SPOD_results.txt','rb') as f:
            results = pickle.load(f)

        Nfft = len(results["freq"])
        Nr = len(results["r"])
        m = -np.fft.fftshift(results["m"])
        Lambdas = np.fft.fftshift(results["lambda"],axes=1)
        Phis = np.fft.fftshift(results["phi"],axes=1)
        
        Ek = np.sum(Lambdas,axis=None)

        _, Ntheta, Nblk = np.shape(Lambdas)
        
        if k==0:
            freqs = results["freq"][0:Nfft//2]*D/U
            Lambdas_mat = np.zeros((len(d_list),Nfft//2,Ntheta,2))
            Phis_mat = np.zeros((len(d_list),len(plot_m),Nr*3))
            # print(f"Nfft = {Nfft}")
            # print(f"Ntheta = {Ntheta}")
        
        Lambdas_mat[k,:,:,:] = Lambdas[:,:,0:2]/Ek
        
       
        mm, ff = np.meshgrid(m,results["freq"][0:Nfft//2])

        for i in range(len(plot_m)):
            m_id = np.argmin(np.abs(m-plot_m[i]))  
            id_max = np.argmax(Lambdas[:,m_id,0]/Lambdas[:,m_id,1])
            print(f"m={int(m[m_id])}, index = {id_max}, St={freqs[id_max].round(3)}")
            id_follow[k,i] = id_max
            Phis_mat[k,i,:] = Phis[freq_selection[i],m_id,:,0]
       
        print("\n")
        if plot_idv:
            fig, ax = plt.subplots()
            norm = clrs.LogNorm(vmin=Lambdas[:,:,0].min()/Ek, vmax=Lambdas[:,:,0].max()/Ek)
            cb = ax.pcolormesh(mm,ff*D/U,Lambdas[:,:,0]/Ek,cmap='hot_r',norm=norm)
            ax.set_xlabel('m')
            ax.set_ylabel('St')
            ax.set_yscale('log')
            ax.set_ylim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            ax.set_xlim(-3,5)
            cbar = fig.colorbar(cb,label=r'first mode energy')
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_spect_zoom',bbox_inches='tight')

            fig, ax = plt.subplots()
            norm = clrs.LogNorm(vmin=Lambdas[:,:,0].min()/Ek, vmax=Lambdas[:,:,0].max()/Ek)
            cb = ax.pcolormesh(mm,ff*D/U,Lambdas[:,:,0]/Ek,cmap='hot_r',norm=norm)
            ax.set_xlabel('m')
            ax.set_ylabel('St')
            ax.set_yscale('log')
            ax.set_ylim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            cbar = fig.colorbar(cb,label=r'first mode energy')
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_spect',bbox_inches='tight')

            fig, ax = plt.subplots()
            norm = clrs.LogNorm(vmin=1, vmax=(Lambdas[:,:,0]/Lambdas[:,:,1]).max())
            cb = ax.pcolormesh(mm,ff*D/U,Lambdas[:,:,0]/Lambdas[:,:,1],cmap='hot_r',norm=norm)
            ax.set_xlabel('m')
            ax.set_ylabel('St')
            ax.set_yscale('log')
            ax.set_xlim(-4,6)
            ax.set_ylim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            cbar = fig.colorbar(cb,label=r'energy separation')
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_separation_zoom',bbox_inches='tight')

            fig, ax = plt.subplots()
            for i in range(Ntheta//2-2,Ntheta//2+3):
                ax.plot(results["freq"][0:Nfft//2]*D/U,Lambdas[:,i,0]/Ek, label=f"m = {m[i]}")
            ax.set_xlabel('St')
            ax.set_ylabel(r'First mode energy')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            ax.legend(fontsize=14)
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_lines',bbox_inches='tight')

            fig, ax = plt.subplots()
            for i in range(Ntheta//2-2,Ntheta//2+3):
                id_max = np.argmax(Lambdas[:,i,0]/Lambdas[:,i,1])
                ax.plot(results["freq"][0:Nfft//2]*D/U,Lambdas[:,i,0]/Lambdas[:,i,1], label=f"m = {m[i]}")
                ax.plot(freqs[id_max],Lambdas[id_max,i,0]/Lambdas[id_max,i,1],ls="",marker=".",c='k')
            ax.set_xlabel('St')
            ax.set_ylabel(r'energy separation')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            ax.legend(fontsize=14)
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_seplines',bbox_inches='tight')

            fig, ax = plt.subplots(figsize=(10,5))  
            colors = plt.cm.cividis((np.linspace(0,1,Nblk)))
            for i in range(Nblk):
                ax.plot(results["freq"][0:Nfft//2]*D/U,np.sum(Lambdas[:,:,i],axis=1)/Ek,c=colors[i])
            ax.plot(results["freq"][0:Nfft//2]*D/U,np.sum(Lambdas,axis=(1,2))/Ek,c='k')
            ax.fill_between(results["freq"][0:Nfft//2]*D/U,np.sum(Lambdas,axis=(1,2))/Ek,np.sum(Lambdas[:,:,0],axis=1)/Ek,color="gainsboro",alpha=0.8)
            ax.set_xlabel('St')
            ax.set_ylabel(r'Sum of m mode energy')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            ax.set_ylim((1e-8,5e-2))
            fig.savefig('./SPODImages/'+fig_name+'_fourier_spod_allm',bbox_inches='tight')
            plt.close('all')
    
    selection_results={'phis': Phis_mat,
                       'm_list': plot_m,
                       'id_freq': freq_selection,
                       'freq': freqs,
                       'r': results["r"]}

    with open('./'+fig_name+'_selected_modes.txt','wb') as f:
            pickle.dump(selection_results,f)

    fig, ax = plt.subplots()
    for j in range(len(plot_m)):
        ax.plot(d_list, id_follow[:,j],ls='',marker='o',label=f'm={plot_m[j]}')
        mode, count = stats.mode(id_follow[:,j], keepdims=False)
        print(f"m={plot_m[j]}, Mode id = {mode}; count = {count}")
    ax.set_xlabel("z")
    ax.set_ylabel("id")
    ax.set_ylim((0,100))
    ax.grid()
    ax.legend()
    fig.savefig('./SPODImages/id_follow',bbox_inches='tight')

  
   
    ff, zz = np.meshgrid(results["freq"][0:Nfft//2],d_list/D/1e3)

    fig, ax = plt.subplots()
    for j in range(len(plot_m)):
        ax.plot(d_list, freqs[id_follow[:,j]],ls='',marker='o',label=f'm={plot_m[j]}')
        mode, count = stats.mode(freqs[id_follow[:,j]], keepdims=False)
        print(f"m={plot_m[j]}, Mode St = {mode}; count = {count}")
    ax.set_xlabel("z")
    ax.set_ylabel("St")
    ax.set_ylim((0,1))
    ax.grid()
    ax.legend()
    fig.savefig('./SPODImages/freq_follow',bbox_inches='tight')
    
    for j in range(Ntheta):
        if m[j] in plot_m:
            fig, ax = plt.subplots()
            cb = ax.contourf(ff*D/U,zz,Lambdas_mat[:,:,j,0],
                             levels=100,
                             cmap='hot_r')
            ax.set_xlabel('St')
            ax.set_ylabel('z')
            ax.set_xscale('log')
            ax.set_xlim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            cbar = fig.colorbar(cb,label=f'm={m[j]} mode energy')
            fig.savefig('./SPODImages/'+case_root[0:8]+'_m'+str(int(m[j]))+'_spod_spect',bbox_inches='tight')

            fig, ax = plt.subplots()
            cb = ax.contourf(ff*D/U,zz,Lambdas_mat[:,:,j,0]/Lambdas_mat[:,:,j,1],
                             levels=np.linspace(1,20,100,endpoint=True),
                             cmap='hot_r',
                             extend='max')
            ax.set_ylabel('z')
            ax.set_xlabel('St')
            ax.set_xscale('log')
            ax.set_xlim((results["freq"][1]*D/U/2,results["freq"][Nfft//2-1]*D/U))
            cb_labels = np.array([1,5,10,15,20])
            cbar = fig.colorbar(cb,label=f'm={m[j]} energy separation')
            cbar.set_ticks(cb_labels)
            cbar.set_ticklabels(cb_labels)
            fig.savefig('./SPODImages/'+case_root[0:8]+'_m'+str(int(m[j]))+'_spod_separation',bbox_inches='tight')


            plt.close('all')
if __name__=="__main__":
    main()
