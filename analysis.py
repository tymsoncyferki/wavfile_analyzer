import numpy as np
from scipy.io import wavfile
from scipy import stats
from config import Config
from scipy.signal import find_peaks

class FileAnalyzer:

    sample_rate: int
    data_raw: np.ndarray
    data: np.ndarray
    frame_size: int
    frames: list
    length: float

    def __init__(self, filepath):
        self.sample_rate, self.data_raw = wavfile.read(filepath)
        self.data = self.to_mono(self.data_raw)
        self.frame_size = int(Config.FRAME_SIZE * self.sample_rate)
        self.frames = [self.data[i:i+self.frame_size] for i in range(0, len(self.data), self.frame_size)][:-1]
        self.length = len(self.data) / self.sample_rate

    @staticmethod
    def to_mono(data) -> np.ndarray:
        """
        Converts dual channel audio to mono
        
        Returns:
            np.ndarray: signal data
        """
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data.astype(np.float32)

    def waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns wave amplitudes and corresponding clip time
        
        Returns:
            tuple[np.ndarray, np.ndarray]: data, time
        """
        time = np.linspace(0, self.length, num=len(self.data))
        
        return self.data, time

    def fft(self, signal=False):
        """
        Computes the FFT of each frame in the input.

        Returns:
            spectrum (np.ndarray): A 2D array of FFT results for each frame.
        """
        if not signal:
            return np.fft.fft(self.frames, axis=1)      
        else:
            return np.fft.fft(self.data)      


    def freq(self, n):
        return np.fft.fftfreq(n, d=1/self.sample_rate)

    def waveform_freq(self):
        """
        
        """
        n = len(self.data)
        fft = np.abs(self.fft(signal=True)) / n
        freqs = self.freq(len(fft))
        half_n = n // 2
        freqs = freqs[:half_n]
        fft = fft[:half_n]
        return fft, freqs

    def volume(self):
        """
        Calculates the volume for each frame using the power of the FFT spectrum.

        Parameters:
            frames (np.ndarray): A 2D array where each row is a signal frame.

        Returns:
            volumes (np.ndarray): A 1D array with the volume for each frame.
        """
        spectra = self.fft()
        power_spectrum = np.abs(spectra) ** 2
        volume = np.mean(power_spectrum, axis=1)
        time = np.linspace(0, self.length, num=len(volume))
        return volume, time


    def ste(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates frames short time energy and corresponding clip time
        
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
            tuple[np.ndarray, np.ndarray]: zcr, time
        """
        zcr = [np.sum(np.abs(np.diff(np.sign(self.data[i:i+self.frame_size])))) / (2 * self.frame_size) for i in range(0, len(self.data), self.frame_size)]
        time = np.linspace(0, self.length, num=len(zcr))

        return zcr, time
    
    def silence(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Checks for silent frames.
        Returns array with value True (or 1) if the frame is silent.

        Returns:
            tuple[np.ndarray, np.ndarray]: silence, time
        """
        volume, time = self.volume()
        zcr, _ = self.zcr()
        
        threshold_volume = np.mean(volume) * 0.5
        threshold_zcr = np.mean(zcr) * 1.5

        silence = np.array((volume < threshold_volume) & (zcr < threshold_zcr))
        
        return silence, time
    
    def silent_ratio(self) -> float:
        """
        Calculates silent ratio

        Returns:
            np.float: silent ratio value
        """
        silence, _ = self.silence()
        return np.mean(silence)

    def f0_frame_autocor(self, signal) -> float:
        """
        Calculates f0 using autocorrelation method for given frame

        Returns:
            float: frame f0
        """
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

    def f0_frame_amdf(self, signal) -> float:
        """
        Calculates f0 using amdf method for given frame

        Returns:
            float: frame f0
        """
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

    def f0(self, method: str = 'cor') -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates fundamental frequency. 
        Available methods: ['cor', 'amdf']

        Returns:
            tuple[np.ndarray, np.ndarray]: f0, time
        """
        if method == 'cor':
            fmethod = self.f0_frame_autocor
        elif method == 'amdf':
            fmethod = self.f0_frame_amdf
        else:
            raise ValueError("method should be in ['cor', 'amdf']")
        
        f0 = [fmethod(self.data[i:i+self.frame_size]**2) for i in range(0, len(self.data), self.frame_size)]
        time = np.linspace(0, self.length, num=len(f0))

        return f0, time

    def voicing(self) -> tuple[list[str], np.ndarray]:
        """
        Calculates voicing of the frames.
        Returns array with 3 values: 'voiced', 'unvoiced' and 'ambigious'

        Returns:
            tuple[np.ndarray, np.ndarray]: labels, time
        """
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
    
    def mv(self) -> float:
        """
        Calculates mean volume
        """
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
    
    def vdr(self) -> float:
        """
        Calculates Volume Dynamic Range
        """
        volume, _ = self.volume()
        return (np.max(volume) - np.min(volume)) / np.max(volume)

    def vu(self) -> float:
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

    def lster(self) -> float:
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

    
    def energy_entropy(self):
        """
        Calculates energy entropy
        """
        K = Config.ENTROPY_K
        sigmas = []
        for i in range(0, len(self.data), self.frame_size):
            frame_data = self.data[i:i+self.frame_size]
            frame_ste = np.mean(frame_data**2) + 0.001

            for k in range(0, len(frame_data), K):
                segment_data = frame_data[k:k+K]
                segment_ste = np.mean(segment_data**2) / frame_ste
                sigmas.append(segment_ste)

        entropy = -np.sum([sigma * np.log2(sigma + 0.001) for sigma in sigmas])

        return entropy

    def rhythm_index(self) -> float:
        """
        Calculates rhythm index
        """
        energy, _ = self.ste()

        peaks, _ = find_peaks(energy, height=np.mean(energy))
        intervals = np.diff(peaks)

        rhythm_score = 1.0 / (np.std(intervals) + 1e-6)
        return rhythm_score

    ## ZCR BASED

    def zstd(self) -> float:
        """
        Calculates standard devation of ZCR
        """
        zcr, _ = self.zcr()
        return np.std(zcr)
    
    def meanzcr(self) -> float:
        """
        Calculates mean ZCR value
        """
        zcr, _ = self.zcr()
        return np.mean(zcr)
    
    def hzcrr(self) -> float:
        """
        Calcualates HZCRR
        """
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
    
    def classify_audio(self):
        """
        Classifies audio into speech and music based on heuristics.
        
        Returns:
            str: audio type 'speech', 'music' or 'other'
        """
        lster = self.lster()
        zcr_std = self.zstd()
        rhythm = self.rhythm_index()
        if np.sum([(lster > 0.3), (zcr_std > 0.07), (rhythm < 0.3)]) >= 2:  # type: ignore
            return "speech"
        elif np.sum([(lster < 0.2), (zcr_std < 0.04), (rhythm > 0.4)]) >= 2:  # type: ignore
            return "music"
        return 'other'
        

    # OTHER

    def calculate_parameters(self):
        params = {}
        params['Audio type'] = self.classify_audio()
        params['Mean Volume'] = self.mv()
        params['Volume STD'] = self.vstd()
        params['Volume Dynamic Range'] = self.vdr()
        params['Volume Undulation'] = self.vu()
        params['Low STE Ratio'] = self.lster()
        params['Energy Entropy'] = self.energy_entropy()
        params['ZCR STD'] = self.zstd()
        params['ZCR Mean'] = self.meanzcr()
        params['Silent Ratio'] = self.silent_ratio()
        params['High ZCR Ratio'] = self.hzcrr()
        params['Rhythm index'] = self.rhythm_index()
        return params
    
    @staticmethod
    def get_stats(array):
        """
        Calculates mean, median, mode
        """
        mean = np.mean(array)
        median = np.median(array)
        mode = stats.mode(array)[0]
        return mean, median, mode
