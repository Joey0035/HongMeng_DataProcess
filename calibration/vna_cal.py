# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import skrf as rf

# def load_calibrator_param(param_file = None):
#     if param_file == None:
#             '''
#     定义标准校准件
#     '''
#     #unit                  L0[H]/C0[F]  L1[H/Hz]/C1[F/Hz]   L2[H/Hz^2]/C2[F/Hz^2]  L3[H/Hz^3]/C3[F/Hz^3]   Delay[s]     Loss [Ω/s]
#     Short_param = np.array([2.0765e-12, -108.54e-24,        2.1705e-33,            -0.01e-42,              31.088e-12,  2.36e9])
#     Open_param  = np.array([49.43e-15,  -310.13e-27,        23.17e-36,             -0.16e-45,              29.243e-12,  2.2e9 ])
#     return Open_param, Short_param

def load_calibrator_param(param_file = None):
    if param_file == None:
            '''
    定义标准校准件
    '''
    #unit                  L0[H]/C0[F]  L1[H/Hz]/C1[F/Hz]   L2[H/Hz^2]/C2[F/Hz^2]  L3[H/Hz^3]/C3[F/Hz^3]   Delay[s]     Loss [Ω/s]
    Short_param = np.array([2.076e-12,   -108.54e-24,        2.171e-33,            -0.01e-42,              31.785e-12,  2.36e9])
    Open_param  = np.array([49.433e-15,  -310.13e-27,        23.168e-36,           -0.16e-45,              29.493e-12,  2.2e9 ])
    return Open_param, Short_param

def get_short_X(f_hz, param):
    p = param[0] + param[1]*f_hz + param[2]*f_hz**2 + param[3]*f_hz**3
    X = 1j*2*np.pi*f_hz*p
    return X

def get_open_X(f_hz, param):
    p = param[0] + param[1]*f_hz + param[2]*f_hz**2 + param[3]*f_hz**3
    X = 1/(1j*2*np.pi*f_hz*p)
    return X

def get_s11_byX(f_hz, param, X):
    Zcl = 50 + (1-1j) * (param[5]/(4*np.pi*f_hz))*np.sqrt(f_hz/1e9)
    alphal = (param[4]*param[5]/100) * (np.sqrt(f_hz/1e9))
    betal = 2*np.pi*f_hz*param[4] + alphal
    gammal = alphal + 1j*betal

    Zin = Zcl*(X+Zcl*np.tanh(gammal))/(Zcl+X*np.tanh(gammal))
    s11 = (Zin - 50) / (Zin + 50)
    return s11

def get_error_param(open, short, load, mo, ms, ml):
    
    S11 = []
    S21 = []
    S22 = []
    for i in range(open.shape[0]):
        A = np.array([[1, open[i], open[i]*mo[i]],[1, short[i], short[i]*ms[i]],[1, load[i], load[i]*ml[i]]])
        b = np.array([mo[i], ms[i], ml[i]])
        x = np.linalg.solve(A,b)
        S11.append(x[0])
        S22.append(x[2])
        S21.append(np.sqrt(x[1]+x[0]*x[2]))
    S11 = np.array(S11)
    S21 = np.array(S21)
    S22 = np.array(S22)

    return S11, S21, S22

def get_cal_S11(S11, S21, S22, S11m):
    S11c = (S11m-S11)/(S21*S21+S22*(S11m-S11))
    return S11c


def vna_cal(
        open_cal,short_cal,load_cal,measure,f_hz,
        cal_measured=False,cal_measured_dir='',cal_measured_prefix='',
        offset=False,offset_dir='',
        plot_s11=False
    ):
    match= load_cal
    short= short_cal
    open1= open_cal
    '''
    将测得的4个数据文件，对幅值和相位做处理，将反射系数写成复数的形式
        dB转未幅值，相位单位由度化为弧度
        以复数的形式把反射系数表示出来
    '''
    s11m=measure
     #=====================================================================
    ml=match    #计算没有校准时，match的反射系数（将 db转化为phasor）   
    ms=short    #计算没有校准时，short的反射系数   
    mo=open1    #计算没有校准时，open的反射系数

    f2=f_hz/1e6     #将VNA测量的频率转化为MHz
    f=f2/1000               #频率由MHz转化为GHz，便于在后面的计算中使用。
    f_hz=f*1e9

    Open_param, Short_param = load_calibrator_param()
    X_short = get_short_X(f_hz, Short_param)
    X_open  = get_open_X(f_hz, Open_param)
    s11_short = get_s11_byX(f_hz,Short_param, X_short)
    s11_open  = get_s11_byX(f_hz,Open_param, X_open)
    s11_load = np.full(f_hz.shape[0],0,dtype=complex)
    if cal_measured:
        s11_open_nw = rf.Network(f'{cal_measured_dir}/{cal_measured_prefix}open.s1p')
        s11_short_nw = rf.Network(f'{cal_measured_dir}/{cal_measured_prefix}short.s1p')
        s11_load_nw = rf.Network(f'{cal_measured_dir}/{cal_measured_prefix}load.s1p')
        s11_open_m = np.polyval(np.polyfit(s11_open_nw.frequency.f/1e6,s11_open_nw.s[:,0,0],2),f2)
        s11_open = np.abs(s11_open) * np.exp(1j*np.angle(s11_open_m))
        s11_short = np.polyval(np.polyfit(s11_short_nw.frequency.f/1e6,s11_short_nw.s[:,0,0],2),f2)
        s11_load = np.polyval(np.polyfit(s11_load_nw.frequency.f/1e6,s11_load_nw.s[:,0,0],2),f2)

    S11, S21, S22 = get_error_param(s11_open, s11_short, s11_load, mo, ms, ml)
    s11_cal = get_cal_S11(S11, S21, S22, s11m)
    
    if offset:
        cable_to_sw12_nw = rf.Network(f'{offset_dir}/cable_port1R60_port2sw12.s2p')
        cable_to_sw4_nw = rf.Network(f'{offset_dir}/cable_port1R60_port2sw4.s2p')
        cable_nw = cable_to_sw4_nw.inv ** cable_to_sw12_nw
        s11_cal_nw = rf.Network(frequency = f_hz, s = s11_cal.reshape(-1,1,1))
        s11_cal = (cable_nw.inv ** s11_cal_nw).s[:,0,0]
    
    # s11_cal= np.concatenate((f_hz[:,None],20.*np.log10(np.abs(s11_cal))[:,None],np.angle(s11_cal,deg=True)[:,None]), axis=1)

    return s11_cal


def _ensure_2d(data):
    data = np.stack(data) if isinstance(data, list) else np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data.astype(complex, copy=False)


def calibrate_vna_sources(
        vna_by_src,
        f_hz,
        switch_cal_keys=None,
        lna_cal_keys=None,
        copy=True,
    ):
    """Batch VNA OSL calibration for a source dictionary.

    Parameters
    ----------
    vna_by_src : dict
        Either ``{name: ndarray}`` or ``{name: {'data': ndarray, ...}}``.
        The returned dictionary stores calibrated data in ``cal_data`` for
        mapping inputs, and returns calibrated arrays for raw-array inputs.
    f_hz : array-like
        VNA frequency axis in Hz.
    switch_cal_keys, lna_cal_keys : dict
        Standard source names, e.g. ``{'open': 'V_Cal_O_H', ...}``.
    copy : bool
        If True, return a shallow-copied dictionary. If False, mutate input.

    Returns
    -------
    (calibrated, diagnostics)
    """
    if switch_cal_keys is None:
        switch_cal_keys = {
            'open': 'V_Cal_O_H',
            'short': 'V_Cal_S_H',
            'load': 'V_Cal_L_H',
        }
    if lna_cal_keys is None:
        lna_cal_keys = {
            'open': 'V_LNA_O_H',
            'short': 'V_LNA_S_H',
            'load': 'V_LNA_L_H',
        }

    def _data(name):
        item = vna_by_src[name]
        return _ensure_2d(item['data'] if isinstance(item, dict) else item)

    out = {}
    diagnostics = {
        'calibrated_sources': [],
        'missing_sources': [],
        'switch_cal_keys': dict(switch_cal_keys),
        'lna_cal_keys': dict(lna_cal_keys) if lna_cal_keys else None,
    }

    for src_name, item in vna_by_src.items():
        data = _data(src_name)
        cal_data = np.empty_like(data)
        use_lna = (
            lna_cal_keys is not None
            and 'LNA' in src_name
            and src_name not in set(lna_cal_keys.values())
        )
        keys = lna_cal_keys if use_lna else switch_cal_keys
        missing = [std for std in keys.values() if std not in vna_by_src]
        if missing:
            diagnostics['missing_sources'].append((src_name, missing))
            cal_data = data
        else:
            open_data = _data(keys['open'])
            short_data = _data(keys['short'])
            load_data = _data(keys['load'])
            n_cycle = min(data.shape[0], open_data.shape[0], short_data.shape[0], load_data.shape[0])
            for i in range(data.shape[0]):
                j = i % n_cycle
                cal_data[i] = vna_cal(open_data[j], short_data[j], load_data[j], data[i], f_hz)
            diagnostics['calibrated_sources'].append(src_name)

        if isinstance(item, dict):
            out_item = dict(item) if copy else item
            out_item['cal_data'] = cal_data
            out[src_name] = out_item
        else:
            out[src_name] = cal_data

    if not copy:
        vna_by_src.update(out)
        return vna_by_src, diagnostics
    return out, diagnostics
