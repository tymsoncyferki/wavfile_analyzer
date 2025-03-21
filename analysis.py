import numpy as np
from scipy.io import wavfile
from config import Config
from scipy.signal import find_peaks

class FileAnalyzer:

    filepath: str
    sample_rate: int
    data_raw: np.ndarray
    data: np.ndarray
    frame_size: int

    def __init__(self, filepath):
        self.sample_rate, self.data_raw = wavfile.read(filepath)
        self.data = self.to_mono(self.data_raw)
        self.frame_size = int(Config.FRAME_SIZE * self.sample_rate)
        self.length = len(self.data) / self.sample_rate

    def to_mono(self, data) -> np.ndarray:
        """
        Converts dual channel audio to mono
        
        Returns:
            np.ndarray: signal data
        """
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data

    def waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns wave amplitudes and coresponding clip time
        
        Returns:
            tuple[np.ndarray, np.ndarray]: data, time
        """
        time = np.linspace(0, self.length, num=len(self.data))
        
        return self.data, time

    def volume(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates frames volume and coresponding clip time
        
        Returns:
            tuple[np.ndarray, np.ndarray]: volume, time
        """
        energy, time = self.ste()
        volume = np.sqrt(energy)
        
        return volume, time

    def ste(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates frames short time    energy and coresponding clip time
        
        Returns:
            tuple[np.ndarray, np.ndarray]: energy, time
        """
        energy = [np.mean(self.data[i:i+self.frame_size]**2) for i in range(0, len(self.data), self.frame_size)]
        time = np.linspace(0, self.length, num=len(energy))
        
        return energy, time
    
    def zcr(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates frames zero crossing rate
        
        Returns:
            np.ndarray: zcr, time
        """
        zcr = [np.sum(np.abs(np.diff(np.sign(self.data[i:i+self.frame_size])))) / (2 * self.frame_size) for i in range(0, len(self.data), self.frame_size)]
        time = np.linspace(0, self.length, num=len(zcr))

        return zcr, time
    
    def silence(self):    
        volume, time = self.volume()
        zcr, _ = self.zcr()
        
        threshold_volume = np.mean(volume) * 0.5
        threshold_zcr = np.mean(zcr) * 1.5

        silence = np.array((volume < threshold_volume) & (zcr < threshold_zcr))
        
        return silence, time
    
    def silent_ratio(self):
        silence, _ = self.silence()
        return np.mean(silence)
    
    def amdf(self, l):
        
        amdf = [np.sum(np.abs(np.diff(self.data[i:i+self.frame_size], l))) for i in range(0, len(self.data), self.frame_size)]    
        time = np.linspace(0, self.length, num=len(amdf))

        return amdf, time

    def f0_frame_autocor(self, signal):
        result = np.correlate(signal, signal, mode='full')
        corr = result[result.size // 2:]  # bierzemy tylko drugą połowę bez laga równego zero

        min_lag = int(self.sample_rate / Config.MAX_F0)  # max F0 = 500 Hz
        max_lag = int(self.sample_rate / Config.MIN_F0)   # min F0 = 50 Hz

        if corr[min_lag:max_lag].size == 0:
            lag = 0
        else:
            lag = np.argmax(corr[min_lag:max_lag]) + min_lag

        f0 = self.sample_rate / lag if lag != 0 else 0
        return f0

    def f0_frame_amdf(self, signal):
        N = len(signal)
        min_lag = int(self.sample_rate / Config.MAX_F0)  # max F0 = 500 Hz
        max_lag = int(self.sample_rate / Config.MIN_F0)   # min F0 = 50 Hz

        amdf_vals = []
        for l in range(min_lag, max_lag):
            diff = np.abs(signal[:-l] - signal[l:])
            amdf_vals.append(np.mean(diff))

        amdf_vals = np.array(amdf_vals)
        lag = np.argmin(amdf_vals) + min_lag
        f0 = self.sample_rate / lag if lag != 0 else 0
        return f0

    def f0(self, method: str = 'cor'):
        if method == 'cor':
            fmethod = self.f0_frame_autocor
        elif method == 'amdf':
            fmethod = self.f0_frame_amdf
        
        f0 = [fmethod(self.data[i:i+self.frame_size]**2) for i in range(0, len(self.data), self.frame_size)]
        time = np.linspace(0, self.length, num=len(f0))

        return f0, time

    def voicing(self):
        zcr, time = self.zcr()
        ste, _ = self.ste()
        
        zcr_thresh = np.median(zcr) 
        ste_thresh = np.median(ste) 

        labels = []

        for ste_i, zcr_i in zip(ste, zcr):
            if ste_i > ste_thresh and zcr_i < zcr_thresh:
                labels.append('voiced')  # ste wysokie, zcr niskie - dźwięczna
            elif ste_i < ste_thresh and zcr_i > zcr_thresh:
                labels.append('unvoiced')  # ste niskie, zcr wysokie - bezdźwięczna
            else:
                labels.append('ambiguous')

        return labels, time
    
    # CLIP

    ## VOLUME BASED PARAMETERS 
    
    def mv(self):
        volume, _ = self.volume()
        return np.mean(volume)

    def vstd(self) -> float:
        """
        Calculates Volume Standard Deviation normalized by the maximum volume in a clip
        """
        volume, _ = self.volume()
        max_vol = np.max(volume)
        std_vol = np.std(volume)
        return std_vol / max_vol
    
    def vdr(self):
        """
        Calculates Volume Dynamic Range
        """
        volume, _ = self.volume()
        return (np.max(volume) - np.min(volume)) / np.max(volume)

    def vu(self):
        """
        Calculates Volume Undulation
        """
        volume, _ = self.volume()
        peaks, _ = find_peaks(volume)
        valleys, _ = find_peaks(-volume)

        extrema_idx = np.sort(np.concatenate((peaks, valleys)))
        extrema_vals = volume[extrema_idx]
        
        diff = np.abs(np.diff(extrema_vals))
        vu_value = np.sum(diff)
        
        return vu_value

    ## ENERGY BASED

    def lster(self):
        """
        Calculates Low STE Rate (LSTER)
        """
        ste, time = self.ste()
        N = len(ste)
        
        frame_step = time[1] - time[0]
        frames_in_1s = int(1.0 / frame_step)

        count = 0

        for n in range(N):
            # window: n - half_window to n + half_window
            half_window = frames_in_1s // 2
            start = max(0, n - half_window)
            end = min(N, n + half_window + 1)
            
            av_ste = np.mean(ste[start:end])

            count += np.sign(0.5 * av_ste - ste[n]) + 1

        lster_value = count / (2 * N)

        return lster_value

    
    def energy_entropy(self, K=100):
    
        sigmas = []
        for i in range(0, len(self.data), self.frame_size):
            frame_data = self.data[i:i+self.frame_size]
            frame_ste = np.mean(frame_data**2)

            for k in range(0, len(frame_data), K):
                segment_data = frame_data[k:k+K]
                segment_ste = np.mean(segment_data**2) / frame_ste
                sigmas.append(segment_ste)

        entropy = -np.sum([sigma * np.log2(sigma) for sigma in sigmas])

        return entropy

    ## ZCR BASED

    def zstd(self):
        zcr, _ = self.zcr()
        return np.std(zcr)
    
    def hzcrr(self):
        zcr, time = self.zcr()
        N = len(zcr)
        
        frame_step = time[1] - time[0]
        frames_in_1s = int(1.0 / frame_step)

        count = 0

        for n in range(N):
            # window: n - half_window to n + half_window
            half_window = frames_in_1s // 2
            start = max(0, n - half_window)
            end = min(N, n + half_window + 1)
            
            av_zcr = np.mean(zcr[start:end])

            count += np.sign(zcr[n] - 1.5 * av_zcr) + 1

        zcr_value = count / (2 * N)

        return zcr_value
    
    # OTHER

    def calculate_parameters(self):
        params = {}
        params['Mean Volume'] = self.mv()
        params['Volume STD'] = self.vstd()
        params['Volume Dynamic Range'] = self.vdr()
        params['Volume Undulation'] = self.vu()
        params['Low STE Ratio'] = self.lster()
        params['Energy Entropy'] = self.energy_entropy()
        params['ZCR STD'] = self.zstd()
        params['Silent Ratio'] = self.silent_ratio()
        params['High ZCR Ratio'] = self.hzcrr()
        return params        
