# -*- coding: utf-8 -*-
"""
Author: JoeyXu
Date: 2025-12-31 10:51:39
Description: 
1. data format convert: .dat to .npz
2. data preprocess
"""

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import rebin
from itertools import cycle,islice
from collections import defaultdict


class DSLpreprocess:
    def __init__(self,file_path,idx_srcsq):
        '''
        file_path: str
            Path to the raw data file.
        idx_srcsq: str
            Index of the source sequence.
        '''
        self.rawdata_dir = file_path
        self.rawdata = np.load(self.rawdata_dir)
        self.smatrix_def = self.sw_def()
        self.obs_sequence = self.gen_obs_sq(idx_srcsq,self.smatrix_def)
    def separation(self):
        '''
        Separate the data into spec and time.
        '''
        spec_sep = {}
        time_sep = {}
        for key in self.data.keys():
            if key.startswith('spec'):
                spec_sep[key] = self.data[key]
            elif key.startswith('time'):
                time_sep[key] = self.data[key]
        return spec_sep,time_sep
    def sw_def(self):
        # =======ver.251216
        # update 0
        s={}
        s[0] = 'V_Ant_H'
        s[1] = 'V_NSon_H'
        s[2] = 'V_NSoff_H'
        s[3] = 'V_HL_H'
        s[4] = 'V_LgO_H'
        s[5] = 'V_LgS_H'
        s[6] = 'V_Cal_L_H'
        s[7] = 'V_Cal_O_H'
        s[8] = 'V_Cal_S_H'
        s[9] = 'V_R3_H'
        s[10] = 'V_R4_H'
        s[11] = 'V_R5_H'
        s[12] = 'V_ShtO_H'
        s[13] = 'V_ShtS_H'
        s[14] = 'V_ShtR1_H'
        s[15] = 'V_ShtR2_H'

        s[20] = 'V_LNAM_H'
        s[21] = 'V_LNA_O_H'
        s[22] = 'V_LNA_S_H'
        s[23] = 'V_LNA_L_H'
        
        s[30] = 'Ant_H'
        s[31] = 'NSon_H'
        s[32] = 'NSoff_H'
        s[33] = 'HL_H'
        s[34] = 'LgO_H'
        s[35] = 'LgS_H'
        s[36] = 'Cal_L_H'
        s[37] = 'Cal_O_H'
        s[38] = 'Cal_S_H'
        s[39] = 'R3_H'
        s[40] = 'R4_H'
        s[41] = 'R5_H'
        s[42] = 'ShtO_H'
        s[43] = 'ShtS_H'
        s[44] = 'ShtR1_H'
        s[45] = 'ShtR2_H'
        return s

    def src_squence(self,idx):
        sq = {}
        # sq['1'] = (1*[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23])
        sq['1'] = (1*[30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
        # sq['2'] = (2*(3*[31,30,32,30]+2*[31,34,35,32,39,40,41,31,30,32])+1*(2*[31,42,43,44,45,32]+1*[31,36,37,38,32])+2*(2*[31,33,30,32,34])+([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23]))
        sq['2'] = (2*(3*[31,30,32,30]+2*[31,34,35,32,39,40,41,31,30,32])+1*(2*[31,42,43,44,45,32]+1*[31,36,37,38,32])+2*(2*[31,33,30,32,34]))
        # sq['3'] = (1*(3*[30,34,35,31,32,33,36]+1*[37,38,31,32,39,40,41,42,43,31,32,44,45])+([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23]))
        sq['3'] = (1*(3*[30,34,35,31,32,33,36]+1*[37,38,31,32,39,40,41,42,43,31,32,44,45]))
        return sq[f'{idx}']

    def gen_obs_sq(self,idx_srcsq,sw_def,int_times=4):
        '''
        Generate observation sequence from source sequence.
        Each source in the source sequence is repeated [int_times] times in the observation sequence.

        Parameters
        ----------
        idx_srcsq : str
            Source sequence index.
        src_def : dict
            Source definition.

        Returns
        -------
        np.ndarray
            Observation sequence.
        '''
        
        src_sq = self.src_squence(idx_srcsq)
        src_sq = [src for src in src_sq for _ in range(int_times)] #Each source repeats four times
        return np.asarray([sw_def[sidx] for sidx in src_sq])


    def data_separation(self,
        obs_seq=None,
        sci_data=None,
        time=None,
        src_offset: int = 0,
        int_times: int = 4
    ):
        """
        Parameters
        ----------
        obs_seq : sequence
            Source sequence (cycled) and repeated [int_times] times.
        sci_data : np.ndarray
            Shape must be (N, 4, 4096) and not rebinned.
        time: np.ndarray
            Shape must be (64*N,)
        src_offset : int
            Source offset (to realign sources)
        int_times : int
            Integration factor of spec.
        Returns
        -------
        dict
            {src: np.ndarray}, each array shaped (M, 4, 4096)
        """
        obs_seq = self.obs_sequence if obs_seq is None else obs_seq
        sci_data = self.rawdata['sci_data'] if sci_data is None else sci_data
        time = self.rawdata['time'] if time is None else time

        if len(obs_seq) == 0:
            raise ValueError("obs_seq must not be empty")
        if sci_data.ndim != 3 or sci_data.shape[1:] != (4, 4096):
            raise ValueError("sci_data must have shape (N, 4, 4096)")
        n_fft = len(sci_data)
        src_start = (src_offset*int_times) % len(obs_seq)

        print(f"{src_start//int_times} Source(s) will be skipped...")
        print(f"Start from source {obs_seq[src_start]}...")

        src_iter = islice(cycle(obs_seq),src_start,src_start+n_fft)

        time = time[::64] # Set each source's first packet time as timestamp

        assert len(time) == n_fft, "Time array must have same length as data array"

        spec_sep = defaultdict(list) # if key not exist, auto create empty {list}.
        time_sep = defaultdict(list) # if key not exist, auto create empty {list}.

        for src,spec,t in zip(src_iter,sci_data,time):
            spec_sep[src].append(spec)
            time_sep[src].append(t)
            
        spec_sep = {k:np.array(v) for k,v in spec_sep.items()}
        time_sep = {k:np.array(v) for k,v in time_sep.items()}
            
        print('-------------Complete----------------')
        print(f"Total {n_fft} spec available, Spec measurement last for about {5*n_fft/60:.2f} min.")
        print(f"Last available source is {src}")
        
        return spec_sep,time_sep
    
    

