# -*- coding: utf-8 -*-
"""
Author: JoeyXu
Date: 2026-01-14 14:27:20
Description: 

"""


from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import rebin
from collections import defaultdict
from datetime import datetime


# =================================================
# ======================Freq=======================
# =================================================
def choose_bw(start_freq,end_freq,samplerate=250,ponits=8192):
    freq_resolution = samplerate/ponits
    idx_start = int(start_freq/freq_resolution)
    idx_end = int(end_freq/freq_resolution)
    return range(idx_start,idx_end,1)




# =================================================
# ============== ======Log-Linear===================
# =================================================

def spec_linear2db(spec):
    return 10.*np.log10(np.abs(spec))

def s11_linear2db(s11):
    return 20.*np.log10(np.abs(s11))





# =================================================
# ======================Time=======================
# =================================================
def toTimeStamp(time_arr,offset=False):
    '''
    toTimeStamp
    time_arr: time array (datetime)
    offset: datetime： datetime(2025,12,18,20,31,0)
    '''
    if offset:
        time_stamp = [offset.timestamp() + t for t in time_arr]
    else:
        time_stamp = [t.timestamp() for t in time_arr]
    return np.asarray(time_stamp)

def toDateTime(time_stamp):
    date_time = list(map(datetime.fromtimestamp, time_stamp))
    return date_time

def choose_time(time_arr,start_time,end_time):
    '''
    choose_time
    time_arr: time array (datetime)
    start_time: datetime
    end_time: datetime
    '''
    idx_start = np.searchsorted(time_arr, start_time)
    idx_end = np.searchsorted(time_arr, end_time)
    return np.asarray(time_arr)[idx_start:idx_end]

def ensure_timestamp(arr):

    if isinstance(arr[0], datetime):
        return np.array([t.timestamp() for t in arr])
    elif isinstance(arr[0], (int, float, np.number)):
        return np.asarray(arr, dtype=float)
    else:
        raise TypeError(f"Unsupported type: {type(arr[0])}")

def get_time_msk(time_arr,chosen_time):
    '''
    :param time_arr: timestamp array
    :param chosen_time: timestamp array
    '''
    time_arr = ensure_timestamp(time_arr)
    chosen_time = ensure_timestamp(chosen_time)
    try:
        time_msk = np.searchsorted(time_arr,chosen_time)
    except:
        print("unsupported input type!")
    return time_msk




# =================================================
# ==================Normalization==================
# =================================================


def max_min_normalization(data):
    norm_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return norm_data