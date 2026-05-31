#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
##-------------------------------

import datetime
import numpy as np
import scipy
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
import time

def gcd(a,b):
    if b!=0:
        return gcd(b,a%b)
    else:
        return a
    
def gcd_in_list(len_list):
    if len(len_list)==1:
        return len_list[0]
    loop = gcd(len_list[0], len_list[1])
    for i in len_list:
        loop = gcd(loop,i)
    return loop

def linuxtime_to_datetime(linux_time):
    '''
    datetime is convenient for plotting
	'''
    date_time = list(map(datetime.datetime.fromtimestamp, linux_time))
    return np.array(date_time)

def rms(data):
    return np.sqrt(np.sum(data**2)/data.shape[0])

def map_complex_to_real(compl_array):
    '''
    Cross correlation should be real theoretically, but it's actually ccomplex.
    There are two ways to map complex to real
    One is get the sign and the absolute value 
    Another is get the real part of the plural
    '''
    # return np.sign(compl_array.real) * np.abs(compl_array)
    return compl_array.real

def freq_range_convert(input_range, freq_stop = 125, freq_step = 16384):
    '''
    Input should be list or np.ndarray which includes two num: start freq and stop freq in Mhz unit.
    e.g. [50,60] represents from 50MHz to 60MHz
    Output is freq index corresponds to the input freq range.
    e.g. [50,60] -----> array([6553, 7863])
    '''
    if(len(input_range) != 2):
        raise Exception('Function freq_range_convert input data length must be 2')
    start_freq = input_range[0]
    stop_freq  = input_range[1]
    start_index = int(start_freq / freq_stop * freq_step)
    stop_index  = int(stop_freq  / freq_stop * freq_step)
    freq_index_range = np.array([start_index , stop_index])
    return freq_index_range

def freq_normalized(freq):
    fr_min = freq[0]
    fr_max = freq[-1]
    fr_mid = (fr_min + fr_max) / 2
    return (freq - fr_mid) / (fr_max - fr_min)

def vis_masked(
    vis_mask,  
    time_mask_index = None, 
    freq_mask_index = None, 
    time_range = None, 
    freq_range = None
):
    '''
    modify the vis_mask by mask_index and range
    '''
    if len(vis_mask.shape) != 2:
        raise Exception('Function vis_masked input data must be 2d.')
    if type(time_mask_index) != type(None):
        vis_mask[time_mask_index, : ] = True
    if type(freq_mask_index) != type(None):
        vis_mask[ :, freq_mask_index] = True
    if type(time_range) != type(None):
        vis_mask[ :time_range[0], : ] = True
        vis_mask[time_range[1]: , : ] = True
    if type(freq_range) != type(None):
        vis_mask[ :, :freq_range[0] ] = True
        vis_mask[ :, freq_range[1]: ] = True
    return vis_mask

def gaussian_smoothing_2d(data,sigma_f=9,sigma_t=12,truncate=3):
    '''smooth the data with gaussian filter'''
    sigma=[sigma_f,sigma_t]
    data_filted=scipy.ndimage.gaussian_filter(data,sigma,truncate=truncate)
    return data_filted

def gaussian_smoothing_1d(data, sigma = 12, truncate = 3):
    date_filted = scipy.ndimage.gaussian_filter1d(data,sigma,truncate = truncate)
    return date_filted

def interpolate_1d(x, y, fill_x, kind = 'slinear'):
    inter_func = interpolate.interp1d(x,y,kind = kind, fill_value = 'extrapolate')
    fill_y = np.array(inter_func(fill_x))
    return fill_y

def interpolate_2d(x1, x2, y, fill_x1, fill_x2, kind = 'linear'):
    inter_func_real = interpolate.interp2d(x1, x2, y.real, kind = kind)
    inter_func_imag = interpolate.interp2d(x1, x2, y.imag, kind = kind)
    fill_y = np.array(inter_func_real(fill_x1, fill_x2)) + 1j*np.array(inter_func_imag(fill_x1, fill_x2))
    # fill_y = griddata((x1,x2),y,(fill_x1,fill_x2),method=kind)
    return fill_y


def S11_convert_2d(S11_VNA, convert_freq):
    '''
    Input S11_VNA:   freq_VNA, amplitude, phase(degree)
    Output S11_spec: S11(freq_spec, real+imag)
    '''
    S11_convert_result = []
    for S11 in S11_VNA:
        S11_convert_result.append(S11_convert_1d(S11,convert_freq))
    return np.array(S11_convert_result)

def S11_convert_1d(S11_VNA, convert_freq):
    '''
    Input S11_VNA:   freq_VNA, amplitude, phase(degree)
    Output S11_spec: S11(freq_spec, real+imag)
    '''
    S11_complex = 10**(S11_VNA[...,1]/20.)*np.exp(1j*np.radians(S11_VNA[...,2]))
    S11_complex = np.insert(S11_complex, 0, 0)
    S11_freq = S11_VNA[...,0]/1e6
    S11_freq = np.insert(S11_freq, 0, 0)
    S11_r = interpolate_1d(S11_freq, S11_complex.real, convert_freq)
    S11_i = interpolate_1d(S11_freq, S11_complex.imag, convert_freq)
    return S11_r + 1j * S11_i

def phase_convert_2(phase):
    for i in range(len(phase[1:])):
        if np.abs(phase[i+1]-phase[i])>=50:
            if phase[i]>=-150:
                phase[i+1:] = phase[i+1:]-180
        if phase[i]<-180:
            phase[i:] = phase[i:]+360
    return phase


def complex_sqrt(data):
    sqrt_amp = np.abs(np.sqrt(data))
    sqrt_phase = phase_convert_2(phase(np.sqrt(data)))
    return sqrt_amp*np.exp(1j*sqrt_phase/180*np.pi)

def data_fit(freq,data,term):
    f = np.polyfit(freq,data,term)
    return np.polyval(f,freq)

def find_closest_time(in_time, check_time):
    in_time2 = np.tile(in_time,(check_time.shape[0],1))
    check_time2 = np.tile(check_time,(in_time.shape[0],1)).T
    return np.abs(in_time2-check_time2).argmin(0)

def S_smooth_1d(S_param, window_length = None, polyorder = None):
    if window_length == None:
        window_length = 11
    if polyorder == None:
        polyorder = 3
    real_smooth = savgol_filter(S_param.real,window_length,polyorder)
    imag_smooth = savgol_filter(S_param.imag,window_length,polyorder)
    S_smooth = real_smooth + 1j*imag_smooth
    return S_smooth

def S_interp_1d(S_param, S_freq, fit_freq):
    S_new = np.interp(fit_freq, S_freq, S_param)
    S_new_mag = np.interp(fit_freq, S_freq, np.abs(S_param))
    S_new_phase = np.angle(S_new)
    S_interp = S_new_mag * np.exp(1j*S_new_phase)

    return S_interp


def cost_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
        return result

    return fun
