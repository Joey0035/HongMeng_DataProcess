import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import sys
from numpy.polynomial import legendre
import os
try:
    from .tools import *
except ImportError:
    from tools import *

class Source(object):

    def __init__(self, freq, t_data, spec_data, Gsrc_data, Grec_data, Tsrc, Tamb, name):
        self.freq = freq       
        self.Gsrc = Gsrc_data
        self.Grec = Grec_data
        self.Ts = np.tile(Tsrc,(len(self.freq),1)).T
        self.Tamb = np.tile(Tamb,(len(self.freq),1)).T
        self.P = spec_data
        self.P_raw = spec_data
        self.Gsrc_raw = Gsrc_data
        self.Grec_raw = Grec_data
        self.t = t_data
        self.name = name

    def average(self):
        
        self.Gsrc = np.mean(self.Gsrc, axis=0)
        self.Grec = np.mean(self.Grec, axis=0)
        self.P = np.mean(self.P, axis=0)
        self.Ts = np.mean(self.Ts, axis=0)
        self.Tamb = np.mean(self.Tamb, axis=0)
        return

    def get_K(self, Gsrc, Grec):

        self.K = {}
        F = np.sqrt(1 - np.abs(Grec)**2)/(1-Gsrc*Grec)
        self.K['ant'] = (1-np.abs(Gsrc)**2) * np.abs(F)**2
        self.K['unc'] = np.abs(Gsrc)**2 * np.abs(F)**2
        self.K['cos'] = (Gsrc * F).real
        self.K['sin'] = (Gsrc * F).imag
        self.K['offset'] = 1
        return
    
    def get_X(self, Pcal, PNS, PL, Gsrc, Grec):
        self.X = {}
        # F = (1-np.abs(Gsrc)**2) / np.abs(1-Gsrc*Grec)**2
        F = 1 / np.abs(1-Gsrc*Grec)**2
        self.X['unc'] = -1 * (np.abs(Gsrc)**2) / np.abs(1-Gsrc*Grec)**2 / F
        self.X['cos'] = -1 * (Gsrc/(1-Gsrc*Grec)).real / np.sqrt(1-np.abs(Grec)**2)  / F
        self.X['sin'] = -1 * (Gsrc/(1-Gsrc*Grec)).imag / np.sqrt(1-np.abs(Grec)**2)  / F
        self.X['ns'] = ((Pcal - PL) / (PNS - PL)) / F
        self.X['L'] = 1 / F

        self.PNS = PNS
        self.PL = PL

        return

    def generate_P(self, Tunc, Tcos, Tsin, Toffset, g=1, add_noise = False):

        P0 = g * (self.K['ant'] * self.Ts + \
                self.K['unc'] * Tunc + \
                self.K['cos'] * Tcos + \
                self.K['sin'] * Tsin + \
                self.K['offset'] * Toffset )
        fr_band = 250/8192
        obs_time = 60*60*100*100
        std = 1/np.sqrt(obs_time * fr_band *1e6)
        if add_noise:
            P = np.random.normal(P0, std*P0, size=P0.shape)
            # P = P
        else:
            P = P0
        return P

class N2_nw(object):
    def __init__(
            self, freq, spec_by_src, vna_by_src, temp_time, temp_data,
            ambient_temp_index=(0, 0), hotload_temp_index=(2, 0),
        ):
        self.freq = freq
        self.spec_by_src = spec_by_src
        self.vna_by_src = vna_by_src
        self.temp_time = temp_time
        self.temp_data = temp_data
        self.ambient_temp_index = tuple(ambient_temp_index)
        self.hotload_temp_index = tuple(hotload_temp_index)

    def _temp_series(self, temp_index):
        if len(temp_index) != 2:
            raise ValueError("temp_index must be a (chip, channel) tuple")
        chip, channel = temp_index
        try:
            return self.temp_data[:, chip, channel]
        except IndexError as exc:
            raise IndexError(
                f"temperature index {temp_index} is outside temp_data shape "
                f"{self.temp_data.shape}"
            ) from exc

    def load_data(self):

        self.src = {}
        self.src_list = []
        ambient_temp = self._temp_series(self.ambient_temp_index)
        hotload_temp = self._temp_series(self.hotload_temp_index)
        for ss_k in self.spec_by_src.keys():
            if 'V_' not in ss_k:
                t_data = self.spec_by_src[ss_k]['time']
                spec_data = self.spec_by_src[ss_k]['data']
                Gsrc_data = self.vna_by_src[f'V_{ss_k}']['cal_data_interpolate']
                Grec_data = self.vna_by_src['V_LNAM_H']['cal_data_interpolate']
                Tamb = self.interpolate_1d(self.temp_time, ambient_temp, self.spec_by_src[ss_k]['time'])
                Tsrc = Tamb
                if 'HL' in ss_k:
                    Tsrc = self.interpolate_1d(self.temp_time, hotload_temp, self.spec_by_src[ss_k]['time'])
                self.src[ss_k[:-2]] = Source(self.freq, t_data, spec_data, Gsrc_data, Grec_data, Tsrc, Tamb, ss_k)
                # self.src[ss_k].average()
                self.src_list.append(ss_k[:-2])
        self.src_list = list(set(self.src_list))


        # for ss_k in self.src.keys():
        #     if ss_k not in ['NSon', 'NSoff']:
        #         self.src[ss_k].get_K(self.src[ss_k].Gsrc, self.src[ss_k].Grec)
        #         self.src[ss_k].get_X(self.src[ss_k].P, self.src['NSon'].P, self.src['NSoff'].P, self.src[ss_k].Gsrc, self.src[ss_k].Grec)
        return

    def interpolate_1d(self, x, y, fill_x, kind = 'slinear'):
        inter_func = interpolate.interp1d(x,y,kind = kind, fill_value = 'extrapolate')
        fill_y = np.array(inter_func(fill_x))
        return fill_y

    def cal_ns_temp(self, src_HL, src_AL, src_Non, src_Noff, TNon_term, TNoff_term, with_nwp = False, ):
        freq = self.freq
        if with_nwp:
            Tunc, Tcos, Tsin = self.Tunc, self.Tcos, self.Tsin
        else:
            Tunc, Tcos, Tsin = np.zeros(len(freq)), np.zeros(len(freq)), np.zeros(len(freq))
        
        Phot = src_HL.P
        Pamb = src_AL.P
        PNon = src_Non.P
        PNoff = src_Noff.P
        Thot = src_HL.Ts
        Tamb = src_AL.Ts
        
        Tnw_hot = src_HL.K['unc'] * Tunc + src_HL.K['cos'] * Tcos + src_HL.K['sin'] * Tsin 
        Tnw_amb = src_AL.K['unc'] * Tunc + src_AL.K['cos'] * Tcos + src_AL.K['sin'] * Tsin
        Tnw_non = src_Non.K['unc'] * Tunc + src_Non.K['cos'] * Tcos + src_Non.K['sin'] * Tsin
        Tnw_noff = src_Noff.K['unc'] * Tunc + src_Noff.K['cos'] * Tcos + src_Noff.K['sin'] * Tsin

        T_offset = ((src_HL.K['ant'] * Thot + Tnw_hot) - Phot/Pamb * (src_AL.K['ant'] * Tamb + Tnw_amb))/(Phot/Pamb - 1)
        # g = Pamb/(src_AL.K['ant'] * Tamb + Tnw_amb + T_offset)
        g = Phot/(src_HL.K['ant'] * Thot + Tnw_hot + T_offset)

        TNon = (PNon/g - Tnw_non - T_offset) / src_Non.K['ant']
        TNoff = (PNoff/g - Tnw_noff - T_offset) / src_Noff.K['ant']

        self.TNon_fbf = TNon-TNoff
        self.TNoff_fbf = TNoff
        self.TNon_poly = legendre.legval(self.freq, legendre.legfit(self.freq, TNon-TNoff, TNon_term))
        self.TNoff_poly = legendre.legval(self.freq, legendre.legfit(self.freq, TNoff, TNoff_term))

        self.T_offset = T_offset
        self.g_Tns = g

        return self.TNon_poly, self.TNoff_poly


    def nwp_fit(self, cal_src_list, TNon, TNoff, fit_term = 7):
        ss_len = int(len(cal_src_list))
        self.nwp_fit_term = fit_term
        self.f_norm = freq_normalized(self.freq)
        self.f_len = int(len(self.f_norm))
        f_len = self.f_len
        A = np.zeros((3 * fit_term, ss_len * f_len))
        b = np.zeros(ss_len * f_len)

        for i,ss_k in enumerate(cal_src_list):
            K = ss_k.K
            Grec = ss_k.Grec
            b[(i + 0)* self.f_len : (i  + 1) * self.f_len] = ss_k.Ts * K['ant'] /(1-np.abs(Grec)**2) - TNon * (ss_k.P - ss_k.PL)/(ss_k.PNS - ss_k.PL) - TNoff
            # b[(i + 0)* self.f_len : (i  + 1) * self.f_len] = ss_k.Ts

            for j in range(fit_term):
                leg_coef = np.zeros(fit_term)
                leg_coef[j] = 1
                A[j + 0 * fit_term, (i + 0) * f_len : (i + 1) * f_len] = -(K['unc'] * legendre.legval(self.f_norm,leg_coef)) /(1-np.abs(Grec)**2)
                A[j + 1 * fit_term, (i + 0) * f_len : (i + 1) * f_len] = -(K['cos'] * legendre.legval(self.f_norm,leg_coef)) /(1-np.abs(Grec)**2)
                A[j + 2 * fit_term, (i + 0) * f_len : (i + 1) * f_len] = -(K['sin'] * legendre.legval(self.f_norm,leg_coef)) /(1-np.abs(Grec)**2)

        M = A.T
        ydata = np.reshape(b, (-1, 1))

        # Solving system using 'short' QR decomposition
        # (see R. Butt, Num. Anal. Using MATLAB)
        Q1, R1 = scipy.linalg.qr(M, mode="economic")
        param = scipy.linalg.solve(R1, np.dot(Q1.T, ydata)).flatten()

        f_X1 = param[0 * fit_term : 1 * fit_term]
        f_X2 = param[1 * fit_term : 2 * fit_term]
        f_X3 = param[2 * fit_term : 3 * fit_term]


        self.f_Tunc = f_X1
        self.f_Tcos = f_X2
        self.f_Tsin = f_X3

        self.Tunc_poly = legendre.legval(self.f_norm,self.f_Tunc)
        self.Tcos_poly = legendre.legval(self.f_norm,self.f_Tcos)
        self.Tsin_poly = legendre.legval(self.f_norm,self.f_Tsin)

        return self.Tunc_poly, self.Tcos_poly, self.Tsin_poly


    def nwp_fbf(self, cal_src_list,  TNon, TNoff):
        ss_len = int(len(cal_src_list))
        self.f_norm = freq_normalized(self.freq)
        self.f_len = int(len(self.f_norm))
        f_len = self.f_len
        A = np.zeros((f_len, 3, ss_len))
        b = np.zeros((f_len, ss_len))

        for i,ss_k in enumerate(cal_src_list):
            K = ss_k.K
            Grec = ss_k.Grec
            A[:,0,i] = -K['unc']
            A[:,1,i] = -K['cos']
            A[:,2,i] = -K['sin']
            # X = ss_k.X
            # A[:,0,i] = X['ns']
            # A[:,1,i] = X['unc']
            # A[:,2,i] = X['cos']
            # A[:,3,i] = X['sin']
            # A[:,4,i] = X['L']

            b[:,i] = ss_k.Ts * K['ant'] /(1-np.abs(Grec)**2) - TNon * (ss_k.P - ss_k.PL)/(ss_k.PNS - ss_k.PL) - TNoff

        theta = []
        for i in range(self.freq.shape[0]):
            x, residuals, rank, s = np.linalg.lstsq(A.transpose((0, 2, 1))[i], b[i], rcond=None)
            theta.append(x)

        theta = np.array(theta)

        self.Tunc_fbf = theta[:,0]
        self.Tcos_fbf = theta[:,1]
        self.Tsin_fbf = theta[:,2]

        return self.Tunc_fbf, self.Tcos_fbf, self.Tsin_fbf


    # def Tsrc_recover(self, Gsrc, X, TNon, Tunc, Tcos, Tsin, TL):
    #     return  (X['ns'] * TNon + X['unc'] * Tunc + X['cos'] * Tcos + X['sin'] * Tsin + X['L'] * TL)/(1-np.abs(Gsrc)**2)
        # return  (X['ns'] * TNon + X['unc'] * Tunc + X['cos'] * Tcos + X['sin'] * Tsin + X['L'] * TL)

    def Tsrc_recover(self, Gsrc, Grec, K, Ps, PNS, PL, TNon, Tunc, Tcos, Tsin, TNoff):
        
        return  (-K['unc'] * Tunc - K['cos'] * Tcos - K['sin'] * Tsin + (1-np.abs(Grec)**2) * (TNon * (Ps - PL)/(PNS - PL) + TNoff))/K['ant']


    def plot_nwp_value(self, freq = None, nwp = None):
        if freq == None:
            freq = self.freq
        if nwp == None:
            TNon, Tunc, Tcos, Tsin, TL= self.TNon, self.Tunc, self.Tcos, self.Tsin, self.TL
            T_hot = np.mean(self.src['HL'].Ts) * np.ones(freq.shape[0])
            T_amb = np.mean(self.src['Cal_L'].Ts) * np.ones(freq.shape[0])
        else:
            TNon, Tunc, Tcos, Tsin, TL = nwp
        fig = plt.figure(figsize=[14,7])
        ax  = fig.add_subplot(2,2,1)
        ax.plot(freq,TNon,label='Noise on temperature')
        # ax.set_xlabel('Freq (MHz)')
        ax.set_ylabel('Noise Souce Temperature')
        ax.grid()
        ax.legend(loc=0)
        ax.set_title('Noise Souce Temperature')
        ax  = fig.add_subplot(2,2,2)
        ax.plot(freq,Tunc,label='Tunc temperature')
        # ax.set_xlabel('Freq (MHz)')
        ax.set_ylabel('Tunc Temperature')
        ax.grid()
        ax.legend(loc=0)
        ax.set_title('Tunc Temperature')

        ax  = fig.add_subplot(2,2,3)
        ax.plot(freq,Tcos, label='Tcos')
        ax.set_xlabel('Freq (MHz)')
        ax.set_ylabel('Tcos Temperature')
        ax.grid()
        ax.legend(loc=0)
        ax.set_title('Tcos Temperature')

        ax  = fig.add_subplot(2,2,4)
        ax.plot(freq,Tsin,label='Tsin')
        ax.set_xlabel('Freq (MHz)')
        ax.set_ylabel('Tsin Temperature')
        ax.grid()
        ax.legend(loc=0)
        ax.set_title('Tsin Temperature')

        plt.figure()
        plt.plot(freq,TL,label='TL')
        plt.grid()
        plt.xlabel('Freq (MHz)')
        plt.ylabel('TL Temperature (K)')
        plt.legend()
        
        return
