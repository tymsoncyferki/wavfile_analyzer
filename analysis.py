import numpy as np
from scipy.io import wavfile
from scipy import stats
from config import Config
from scipy.signal import find_peaks
from scipy.signal.windows import get_window
from scipy.fft import rfft, rfftfreq

class FileAnalyzer:

    sample_rate: int
    data_raw: np.ndarray
    data: np.ndarray
    frame_size: int
    length: float

    def __init__(self, filepath):
        self.sample_rate, self.data_raw = wavfile.read(filepath)
        self.data = self.to_mono(self.data_raw)
        self.frame_size = int(Config.FRAME_SIZE * self.sample_rate)
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

    def frames(self):
        """ returns frames """
        return [self.data[i:i+self.frame_size] for i in range(0, len(self.data), self.frame_size)][:-1]
    
    def time(self, n):
        """ returns n time values """
        return np.linspace(0, self.length, num=n)

    def waveform(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns wave amplitudes and corresponding clip time
        
        Returns:
            tuple[np.ndarray, np.ndarray]: data, time
        """
        time = np.linspace(0, self.length, num=len(self.data))
        
        return self.data, time

    def fft(self, data=None, frames=False) -> np.ndarray:
        """
        Computes the FFT of the input

        Args:
            frames (bool): if true computes the FFT of each frame in the input

        Returns:
            spectrum (np.ndarray): an array of FFT results (2d with fft for each frame if frames=true)
        """
        if data is None:
            if frames:
                data = self.frames()
            else:
                data = self.data
        if frames:
            return rfft(data, axis=1)  # type: ignore
        else:
            return rfft(data)  # type: ignore

    def freq(self, n=None, frames=False):
        """
        Generates fft frequency for data of length n
        """
        if n is None:
            if frames:
                n = len(self.frames()[0])
            else:
                n = len(self.data)
        return rfftfreq(n, d=1/self.sample_rate)

    def get_band_indices(self, freqs):
        fs = self.sample_rate
        ratio = fs / 22050
        base_bands = [
            (0, 630),
            (630, 1720),
            (1720, 4400),
            (4400, 11025)
        ]
        bands = [(low * ratio, high * ratio) for (low, high) in base_bands]
        band_indices = [np.where((freqs >= low) & (freqs < high))[0] for (low, high) in bands]
        return band_indices


    def waveform_freq(self, data=None, normalize=False):
        """
        Calculates fft and returns it with corresponding frequency. Returns only positive values

        Args:
            data (np.ndarray)
            normalize (bool): if to normalize the fft

        Returns:
            fft, freqs
        """
        if data is None:
            data = self.data

        n = len(data)
        fft = np.abs(self.fft(data=data))
        if normalize:
            fft = fft / n
        freqs = self.freq(len(data))

        return fft, freqs

    def volume(self):
        """
        Calculates the volume for each frame using the power of the FFT spectrum.

        Parameters:
            frames (np.ndarray): A 2D array where each row is a signal frame.

        Returns:
            volumes (np.ndarray): A 1D array with the volume for each frame.
        """
        spectra = self.fft(frames=True)
        power_spectrum = np.abs(spectra) ** 2
        volume = np.mean(power_spectrum, axis=1)
        time = self.time(len(volume))
        return volume, time
    
    @staticmethod
    def apply_window(data, method='hann'):
        """
        Applies window

        Args:
            data: np.ndarray
            method: window method to use

        """
        N = len(data)
        window = get_window(method, N)
        return data * window
    

    def fc(self):
        """
        Calculates Frequency Centroid for each frame

        Returns:
            centroids, time
        """
        # REPORT - DISCRETE VERSION
        
        spectrums = self.fft(frames=True)
        freqs = self.freq(frames=True)
        
        magnitudes = np.abs(spectrums)

        centroids = [np.sum(freqs * mag) / np.sum(mag) if np.sum(mag) != 0 else 0 for mag in magnitudes]
        time = self.time(len(centroids))
        
        return centroids, time

    def bw(self):
        """
        Calculates effective bandwidth
        
        Returns:
            bandwidth, time
        """
        spectrums = self.fft(frames=True)
        freqs = self.freq(frames=True)
        magnitudes = np.abs(spectrums)
        centroids, _ = self.fc()

        bandwidths = [np.sqrt(np.sum(((freqs - fc) ** 2) * mag) / np.sum(mag)) if np.sum(mag) != 0 else 0 for mag, fc in zip(magnitudes, centroids)]
        time = self.time(len(bandwidths))
        return bandwidths, time
    
    def ersb(self):
        """
        Calculates Band Energy Ratio (ERSB) for each frame with subband boundaries scaled to current sample rate

        Returns:
            ersb1, ersb2, ersb3, time
        """
        spectrums = self.fft(frames=True)
        freqs = self.freq(frames=True)
        powers = np.abs(spectrums) ** 2

        band_indices = self.get_band_indices(freqs)

        ersb1 = []
        ersb2 = []
        ersb3 = []

        for frame_power in powers:
            total_energy = np.sum(frame_power)
            if total_energy == 0:
                ersb1.append(0)
                ersb2.append(0)
                ersb3.append(0)
            else:
                # we use those powers which are for frequcneis in i-th bandwidth
                b1 = np.sum(frame_power[band_indices[0]]) / total_energy  # divide energy of band by whole energy
                b2 = np.sum(frame_power[band_indices[1]]) / total_energy
                b3 = np.sum(frame_power[band_indices[2]]) / total_energy
                ersb1.append(b1)
                ersb2.append(b2)
                ersb3.append(b3)

        time = self.time(len(ersb1))
        return ersb1, ersb2, ersb3, time

    def sfm(self):
        """
        Calculates spectral flatness measure for each frame

        Returns:
            sfm, time
        """
        spectrums = self.fft(frames=True)
        powers = np.abs(spectrums) ** 2

        sfm_list = []

        for mag in powers:
            if np.all(mag == 0):
                sfm_list.append(1.0)
            else:
                geo_mean = np.exp(np.mean(np.log(mag + 1e-12)))  # dodajemy epsilon, żeby uniknąć log(0)
                arith_mean = np.mean(mag)
                sfm = geo_mean / arith_mean
                sfm_list.append(sfm)

        time = self.time(len(sfm_list))
        return sfm_list, time


    def sfm_bands(self):
        """
        Calculates spectral flatness measure (SFM) for subbands

        Returns:
            sfm_b1, sfm_b2, sfm_b3, time
        """
        spectrums = self.fft(frames=True)
        freqs = self.freq(frames=True)
        powers = np.abs(spectrums) ** 2

        band_indices = self.get_band_indices(freqs)

        sfm_b1, sfm_b2, sfm_b3 = [], [], []

        for frame_power in powers:
            for sfm_list, indices in zip([sfm_b1, sfm_b2, sfm_b3], band_indices[:3]):
                values = frame_power[indices]
                if np.all(values == 0):
                    sfm = 1.0
                else:
                    geo_mean = np.exp(np.mean(np.log(values + 1e-12)))  # adding epsilon to prevent log(0)
                    arith_mean = np.mean(values)
                    sfm = geo_mean / arith_mean
                sfm_list.append(sfm)

        time = self.time(len(sfm_b1))
        return sfm_b1, sfm_b2, sfm_b3, time

    
    def scf_bands(self):
        """
        Calculates Spectral Crest Factor (SCF) for subbands in each frame.

        Returns:
            scf_b1, scf_b2, scf_b3, time
        """
        spectrums = self.fft(frames=True)
        freqs = self.freq(frames=True)
        powers = np.abs(spectrums) ** 2

        band_indices = self.get_band_indices(freqs)

        scf_b1, scf_b2, scf_b3 = [], [], []

        for frame_power in powers:
            for scf_list, indices in zip([scf_b1, scf_b2, scf_b3], band_indices[:3]):
                values = frame_power[indices]
                if np.all(values == 0):
                    scf = 0.0
                else:
                    peak = np.max(values)
                    mean = np.mean(values)
                    scf = peak / mean
                scf_list.append(scf)

        time = self.time(len(scf_b1))

        return scf_b1, scf_b2, scf_b3, time

    
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
