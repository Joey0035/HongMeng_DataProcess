"""
HongMeng Data Inspector
=======================
数据检视工具：从 HongMeng_raw_data_Parser 解析结果中提取、分割、可视化数据。

Classes
-------
EffectiveDataExtractor
    从完整解析结果中提取有效子集（data / time / src）。
DataInspector
    按源分割 SPEC / VNA 数据，打印 info，绘制频谱 / S11 面板。

Usage
-----
    from HongMeng_DataInspector import EffectiveDataExtractor, DataInspector
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ======================================================================
#  EffectiveDataExtractor
# ======================================================================

class EffectiveDataExtractor:
    """从解析结果中提取有效数据子集。

    Parameters
    ----------
    data_parsed : dict
        HongMengFileProcessor.process_file() 的完整返回结果。
    mode : str
        提取模式。目前仅支持 ``"Minimal"``。

    Minimal 模式输出
    ----------------
    eff_data['spec'] / ['vna'] = {'data': ndarray, 'time': ndarray, 'src': ndarray}
    """

    def __init__(self, data_parsed: dict, mode: str = "Minimal"):
        self.data_parsed = data_parsed
        self.mode = mode

    def extract_effective_data(self) -> dict:
        if self.mode == "Minimal":
            eff_data = {}
            for key in ['spec', 'vna']:
                if key in self.data_parsed:
                    eff_data[key] = {
                        'data': self.data_parsed[key]['data'],
                        'time': self.data_parsed[key]['time'],
                        'src':  self.data_parsed[key]['src'],
                    }
            assert (eff_data['spec']['data'].shape[0]
                    == eff_data['spec']['time'].shape[0] / 64
                    == eff_data['spec']['src'].shape[0] / 64), \
                "Spec data, time and src length mismatch"
            return eff_data
        raise NotImplementedError(f"Mode {self.mode} not implemented yet.")


# ======================================================================
#  DataInspector
# ======================================================================

class DataInspector:
    """对设备运行状态进行检查

    Parameters
    ----------
    data_parsed : dict
        HongMengFileProcessor.process_file() 的完整返回结果。
    vna_freq : dict, optional
        VNA 频率配置，``{n_freq: (start_mhz, stop_mhz)}``。
        例如 ``{901: (30, 120), 1901: (1, 190)}``。
        未提供时瀑布图 x 轴用频点索引。
    """

    _PKTS_PER_FFT = 64
    _SPEC_BW_MHZ  = 250.0
    _SPEC_N_CH    = 4096
    _DTYPE_LABEL  = {
        '1': 'Auto1', '2': 'Auto2',
        'r': 'Cross-Real', 'i': 'Cross-Imag',
    }

    def __init__(self, data_parsed: dict, vna_freq: Optional[Dict[int, Tuple[float, float]]] = None):
        self.data     = data_parsed
        self.src_def  = self._build_src_def()
        self.vna_freq = vna_freq or {}
        self._cache: dict = {}

    # ================================================================
    #  Source definitions
    # ================================================================

    @staticmethod
    def _build_src_def() -> dict:
        s: dict = {}
        base = ['Ant', 'NSon', 'NSoff', 'HL', 'LgO', 'LgS',
                'Cal_L', 'Cal_O', 'Cal_S', 'R3', 'R4', 'R5',
                'ShtO', 'ShtS', 'ShtR1', 'ShtR2']
        for i, name in enumerate(base):
            s[i]      = f'V_{name}_H'
            s[i + 30] = f'{name}_H'
        s[20], s[21], s[22], s[23] = 'V_LNAM_H', 'V_LNA_O_H', 'V_LNA_S_H', 'V_LNA_L_H'
        return s

    def _src_name(self, sid) -> str:
        return self.src_def.get(int(sid), f'src_{sid}')

    @staticmethod
    def _ri12_map(dtype: str) -> int:
        """'1'->0 (Auto1), '2'->1 (Auto2), 'r'->3 (Cross-Real), 'i'->2 (Cross-Imag)"""
        m = {'1': 0, '2': 1, 'r': 3, 'i': 2}
        if dtype not in m:
            raise ValueError(f"dtype must be '1','2','r','i', got '{dtype}'")
        return m[dtype]

    # ================================================================
    #  Frequency helpers
    # ================================================================

    def _spec_freq_mhz(self) -> np.ndarray:
        """SPEC 频率轴 (MHz)"""
        return np.linspace(0, self._SPEC_BW_MHZ, self._SPEC_N_CH)

    def _vna_freq_mhz(self, n_freq: int) -> Optional[np.ndarray]:
        """VNA 频率轴 (MHz)；若无配置返回 None"""
        if n_freq in self.vna_freq:
            f0, f1 = self.vna_freq[n_freq]
            return np.linspace(f0, f1, n_freq)
        return None

    # ================================================================
    #  Internal helpers
    # ================================================================

    def _fft_src_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 per-FFT 的 (src, time) 数组"""
        spec  = self.data['spec']
        n_fft = spec['data'].shape[0]
        step  = self._PKTS_PER_FFT
        return spec['src'][::step][:n_fft], spec['time'][::step][:n_fft]

    def _vna_sweep_table(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 per-sweep 的 (src_id, n_freq) 数组"""
        vna = self.data['vna']
        gf  = vna['metadata']['group_flag']
        sweep_starts = np.where(gf == 1)[0]
        n_sweep = vna['data'].shape[0]
        return vna['src'][sweep_starts[:n_sweep]], vna['n_freq_per_sweep'][:n_sweep]

    def _ordered_sids(self, sub_key: str, unique_src: np.ndarray) -> list:
        """按 obs_seq 排序的 sid 列表，不在 obs_seq 中的附加末尾"""
        obs = self.data[sub_key].get('obs_seq', np.array([]))
        ordered = list(obs) if len(obs) else sorted(unique_src)
        extra = [s for s in unique_src if int(s) not in [int(x) for x in ordered]]
        return list(ordered) + extra

    # ================================================================
    #  Split methods
    # ================================================================

    def split_spec_by_src(self) -> Dict[str, np.ndarray]:
        """按源分割频谱 → {src_name: (n_fft_src, 4, 4096)}"""
        if 'spec' in self._cache:
            return self._cache['spec']

        data = self.data['spec']['data']
        fft_src, _ = self._fft_src_time()
        sids = self._ordered_sids('spec', np.unique(fft_src))

        result = {}
        for sid in sids:
            mask = fft_src == int(sid)
            if mask.any():
                result[self._src_name(sid)] = data[mask]

        self._cache['spec'] = result
        return result

    def split_vna_by_src(self, n_freq: Optional[int] = None) -> dict:
        """按源分割 VNA 扫频

        Parameters
        ----------
        n_freq : int, optional
            仅返回指定频点数的扫频。未指定时返回所有（混合长度用 list）。

        Returns
        -------
        dict : {src_name: ndarray (n_sweep_src, n_freq) 或 list}
        """
        cache_key = f'vna_{n_freq}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        vna = self.data['vna']
        s11 = vna['data']
        sweep_src, sweep_nf = self._vna_sweep_table()

        freq_mask = (sweep_nf == n_freq) if n_freq is not None else np.ones(len(sweep_src), dtype=bool)
        filtered_src = sweep_src[freq_mask]
        sids = self._ordered_sids('vna', np.unique(filtered_src))

        result = {}
        for sid in sids:
            src_mask = (sweep_src == int(sid)) & freq_mask
            if not src_mask.any():
                continue
            name = self._src_name(sid)
            indices = np.where(src_mask)[0]
            if s11.dtype == object:
                sweeps = [s11[i] for i in indices]
                try:
                    result[name] = np.stack(sweeps)
                except ValueError:
                    result[name] = sweeps
            else:
                result[name] = s11[indices]

        self._cache[cache_key] = result
        return result

    def split_spec_time_by_src(self, include_time: bool = False) -> Dict[str, np.ndarray]:
        """按源分割频谱 → {src_name: (n_fft_src, 4, 4096)} 或 {src_name: {'data': ..., 'time': ...}}

        Parameters
        ----------
        include_time : bool, optional
            若为 True，返回 {src_name: {'data': ndarray, 'time': ndarray}}
            若为 False（默认），返回 {src_name: ndarray}
        """
        cache_key = 'spec_with_time' if include_time else 'spec'
        # if cache_key in self._cache:
        #     return self._cache[cache_key]

        data = self.data['spec']['data']
        fft_src, fft_time = self._fft_src_time()
        sids = self._ordered_sids('spec', np.unique(fft_src))

        result = {}
        for sid in sids:
            mask = fft_src == int(sid)
            if mask.any():
                name = self._src_name(sid)
                if include_time:
                    result[name] = {
                        'data': data[mask],
                        'time': fft_time[mask]
                    }
                else:
                    result[name] = data[mask]

        self._cache[cache_key] = result
        return result

    def split_vna_time_by_src(self, n_freq: Optional[int] = None, include_time: bool = False) -> dict:
        """按源分割 VNA 扫频

        Parameters
        ----------
        n_freq : int, optional
            仅返回指定频点数的扫频。未指定时返回所有（混合长度用 list）。
        include_time : bool, optional
            若为 True，返回 {src_name: {'data': ndarray, 'time': ndarray}}
            若为 False（默认），返回 {src_name: ndarray}

        Returns
        -------
        dict : {src_name: ndarray (n_sweep_src, n_freq) 或 list} 或带 time 的字典
        """
        cache_key = f'vna_{n_freq}_{"with_time" if include_time else "no_time"}'
        # if cache_key in self._cache:
        #     return self._cache[cache_key]

        vna = self.data['vna']
        s11 = vna['data']
        sweep_src, sweep_nf = self._vna_sweep_table()

        freq_mask = (sweep_nf == n_freq) if n_freq is not None else np.ones(len(sweep_src), dtype=bool)
        filtered_src = sweep_src[freq_mask]
        sids = self._ordered_sids('vna', np.unique(filtered_src))

        result = {}
        for sid in sids:
            src_mask = (sweep_src == int(sid)) & freq_mask
            if not src_mask.any():
                continue
            name = self._src_name(sid)
            indices = np.where(src_mask)[0]

            if s11.dtype == object:
                sweeps = [s11[i] for i in indices]
                try:
                    data_array = np.stack(sweeps)
                except ValueError:
                    data_array = sweeps
            else:
                data_array = s11[indices]

            if include_time:
                # 获取对应的时间
                sweep_starts = np.where(vna['metadata']['group_flag'] == 1)[0]
                n_sweep = vna['data'].shape[0]
                sweep_time = vna['time'][sweep_starts[:n_sweep]]
                result[name] = {
                    'data': data_array,
                    'time': sweep_time[indices]
                }
            else:
                result[name] = data_array

        self._cache[cache_key] = result
        return result


    # ================================================================
    #  Info
    # ================================================================

    def info(self):
        """打印数据概览：shape / sources / obs_seq / VNA 频点分组"""
        for key in ('spec', 'vna', 'temp'):
            if key not in self.data:
                continue
            sub = self.data[key]
            d = sub['data']
            shape = d.shape if hasattr(d, 'shape') else '?'
            srcs = np.unique(sub['src'])
            print(f"[{key.upper()}]  data: {shape}  |  {len(srcs)} sources")
            if 'obs_seq' in sub:
                seq = [self._src_name(s) for s in sub['obs_seq']]
                print(f"  obs_seq ({len(seq)}): {seq}")
            if key == 'vna' and 'n_freq_per_sweep' in sub:
                for nf in np.unique(sub['n_freq_per_sweep']):
                    cnt = np.sum(sub['n_freq_per_sweep'] == nf)
                    freq_info = ''
                    if nf in self.vna_freq:
                        f0, f1 = self.vna_freq[nf]
                        freq_info = f' ({f0}-{f1} MHz)'
                    print(f"  n_freq={nf}{freq_info}: {cnt} sweeps")
            print()

    # ================================================================
    #  Plotting — SPEC
    # ================================================================

    def plot_spec_panel(self, dtype: str = '1'):
        """频谱面板 (MHz / dB)

        Parameters
        ----------
        dtype : '1' (Auto-A) | '2' (Auto-B) | 'r' (Cross-Imag) | 'i' (Cross-Real)

        Returns
        -------
        fig1 (first cycle), fig2 (mean), fig3 (waterfall)
        """
        ch    = self._ri12_map(dtype)
        label = self._DTYPE_LABEL[dtype]
        split = self.split_spec_by_src()
        src_order = list(split.keys())
        n_src = len(src_order)
        freq  = self._spec_freq_mhz()

        def to_dB(arr):
            return 10 * np.log10(np.abs(arr).clip(1e-30))

        # --- Fig 1: First cycle spectra ---
        fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=120)
        for name in src_order:
            ax1.plot(freq, to_dB(split[name][0, ch, :]), label=name, lw=0.7)
        ax1.set(title=f'First Cycle Spectra \u2014 {label}',
                xlabel='Frequency (MHz)', ylabel='Power (dB)')
        ax1.legend(fontsize=7, ncol=3, loc='upper right')
        ax1.grid(alpha=0.3)
        fig1.tight_layout()

        # --- Fig 2: Mean spectra ---
        fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=120)
        for name in src_order:
            ax2.plot(freq, to_dB(split[name][:, ch, :].mean(axis=0)), label=name, lw=0.7)
        ax2.set(title=f'Mean Spectra (all cycles) \u2014 {label}',
                xlabel='Frequency (MHz)', ylabel='Power (dB)')
        ax2.legend(fontsize=7, ncol=3, loc='upper right')
        ax2.grid(alpha=0.3)
        fig2.tight_layout()

        # --- Fig 3: Waterfall per source ---
        ncols = min(4, n_src)
        nrows = -(-n_src // ncols)
        fig3, axes = plt.subplots(nrows, ncols,
                                  figsize=(4 * ncols, 3 * nrows), dpi=100,
                                  squeeze=False)
        for idx, name in enumerate(src_order):
            ax = axes[idx // ncols, idx % ncols]
            wf = to_dB(split[name][:, ch, :])
            im = ax.pcolormesh(wf, shading='auto', rasterized=True)
            n_ticks = 5
            tick_pos = np.linspace(0, wf.shape[1] - 1, n_ticks)
            tick_lbl = [f'{v:.0f}' for v in np.linspace(freq[0], freq[-1], n_ticks)]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel('Freq (MHz)', fontsize=7)
            ax.set_ylabel('FFT #', fontsize=7)
            ax.tick_params(labelsize=6)
            fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for idx in range(n_src, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig3.suptitle(f'Waterfall \u2014 {label}  (dB)', fontsize=12)
        fig3.tight_layout()
        plt.show()
        return fig1, fig2, fig3

    # ================================================================
    #  Plotting — VNA
    # ================================================================

    def plot_vna_panel(self, n_freq: Optional[int] = None):
        """VNA S11 面板 (MHz / dB)

        Parameters
        ----------
        n_freq : int, optional
            指定频点数（如 901 / 1901）。未指定时自动按频点数分组画图。

        Returns
        -------
        list of (fig1, fig2, fig3) per frequency configuration
        """
        _, sweep_nf = self._vna_sweep_table()
        nf_groups = [n_freq] if n_freq else sorted(np.unique(sweep_nf))
        all_figs: List[Tuple] = []

        for nf in nf_groups:
            split = self.split_vna_by_src(n_freq=nf)
            if not split:
                continue
            src_order = list(split.keys())
            n_src = len(src_order)

            freq_ax = self._vna_freq_mhz(nf)
            x      = freq_ax if freq_ax is not None else np.arange(nf)
            xlabel = 'Frequency (MHz)' if freq_ax is not None else 'Freq Index'
            tag    = f'{nf} pts'
            if freq_ax is not None:
                tag += f' ({freq_ax[0]:.0f}-{freq_ax[-1]:.0f} MHz)'

            def s11_dB(s):
                return 20 * np.log10(np.abs(s).clip(1e-30))

            # --- Fig 1: First sweep per source ---
            fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=120)
            for name in src_order:
                d = split[name]
                first = d[0] if isinstance(d, np.ndarray) else d[0]
                ax1.plot(x, s11_dB(first), label=name, lw=0.7)
            ax1.set(title=f'VNA S11 \u2014 First Sweep \u2014 {tag}',
                    xlabel=xlabel, ylabel='|S11| (dB)')
            ax1.legend(fontsize=7, ncol=3, loc='upper right')
            ax1.grid(alpha=0.3)
            fig1.tight_layout()

            # --- Fig 2: Mean S11 per source ---
            fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=120)
            for name in src_order:
                d = split[name]
                if isinstance(d, np.ndarray) and d.ndim == 2:
                    mean_s11 = np.abs(d).mean(axis=0)
                else:
                    mean_s11 = np.abs(d[0])
                ax2.plot(x, 20 * np.log10(mean_s11.clip(1e-30)),
                         label=name, lw=0.7)
            ax2.set(title=f'VNA S11 \u2014 Mean \u2014 {tag}',
                    xlabel=xlabel, ylabel='|S11| (dB)')
            ax2.legend(fontsize=7, ncol=3, loc='upper right')
            ax2.grid(alpha=0.3)
            fig2.tight_layout()

            # --- Fig 3: Waterfall per source ---
            ncols = min(4, n_src)
            nrows = -(-n_src // ncols)
            fig3, axes = plt.subplots(nrows, ncols,
                                      figsize=(4 * ncols, 3 * nrows), dpi=100,
                                      squeeze=False)
            for idx, name in enumerate(src_order):
                ax = axes[idx // ncols, idx % ncols]
                d  = split[name]
                if isinstance(d, np.ndarray) and d.ndim == 2:
                    wf = s11_dB(d)
                else:
                    wf = s11_dB(np.stack(d) if len(d) > 1 else np.atleast_2d(d[0]))
                im = ax.pcolormesh(wf, shading='auto', rasterized=True)
                if freq_ax is not None:
                    n_ticks = 5
                    tick_pos = np.linspace(0, wf.shape[1] - 1, n_ticks)
                    tick_lbl = [f'{v:.0f}' for v in np.linspace(freq_ax[0], freq_ax[-1], n_ticks)]
                    ax.set_xticks(tick_pos)
                    ax.set_xticklabels(tick_lbl)
                ax.set_title(name, fontsize=8)
                ax.set_xlabel(xlabel.replace('Frequency', 'Freq'), fontsize=7)
                ax.set_ylabel('Sweep #', fontsize=7)
                ax.tick_params(labelsize=6)
                fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for idx in range(n_src, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig3.suptitle(f'VNA Waterfall \u2014 {tag}  (dB)', fontsize=12)
            fig3.tight_layout()
            plt.show()
            all_figs.append((fig1, fig2, fig3))

        return all_figs
