##Script to store all the Sample Detector helper functions

import numpy as np
import librosa
from scipy.signal import find_peaks

class Fingerprint:
    def __init__(self, song_url):
        self.song_url = song_url
        self.time_series, self.sampling_rate = librosa.load(song_url)
        self.n_fft = 2048
        self.hop_length = 512
        self.sftf = np.abs(librosa.stft(self.time_series, hop_length=512))
        
    '''
        Funtion to return the index locations from the output of librosa.sftf that correspond to frequencies 
        between what the human ear can hear:
            20Hz - 6000Hz
        
        Since the sampling rate can vary for each song there is a 5Hz buffer on either side of the target frequency
        to ensure we capture the correct bin
    '''
    def get_freq_loc(self):
        frequency_bins = np.arange(0,1 + self.n_fft/2) * self.sampling_rate/self.n_fft 
        human_hearing_min, human_hearing_max = 20, 6000
        locations = [] 
        for i, f in enumerate(frequency_bins): 
            if f >= human_hearing_min and f <= human_hearing_max:
                locations.append(i) 
        return locations
    
    '''
        Function to extract events from the spectrogram based on peak detection
        
        Will filter the output of sftf to only look at bins of interest and then will find the maximum values
        within each frequency band that exceed a certain threshold (height >= .6 by default)
        
        Will then return a list of tuples where each tuple contains the frequency bin index and the time step index
        corresponding to an event detected in the spectrogram. The list will be sorted based on the recorded time step
    '''
    def extract_events(self):
        events = []
        
        locs = self.get_freq_loc()
        band_energy = self.sftf[locs, :]

        for band in range(band_energy.shape[0]):
            thresh = np.percentile(band, 90)
            peaks_loc, _ = find_peaks(band_energy[band], height=thresh)
            for t in peaks_loc:
                #for each event it is storing the index corresponding to a frequency band and the time step
                events.append((locs[band], t))
                
        #sorting the events based on time step
        events = sorted(events, key=lambda x: x[1])
        return events
    
    '''
        Function to use the events extracted to build a fingerprint of the song
        
        Will set an anchor point (b1, t1) and then look for other points (b2, t2) that occur after
        Will only store the difference between the anchor and target if is less than or equal to some maximum time delta
        default is set at 6 time steps
        
        The final output will be a list of tuples that stores the hash: anchor, target, and the time delta, and the time stamp
        ex. ((b1, b2, dt), t1) -> (hash, time stamp)
    '''
    def build_fingerprint(self, max_dt=6):
        fingerprints = []
        
        events = self.extract_events()

        for i, (b1, t1) in enumerate(events):
            for b2, t2 in events[i+1:]:
                dt = t2 - t1
                if 0 < dt <= max_dt:
                    fingerprints.append(((b1, b2, dt),t1))
                if dt > max_dt:
                    break

        return fingerprints
