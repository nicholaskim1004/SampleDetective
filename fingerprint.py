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
        
        To help with creating a sparse fingerprint will group the frequency bins into bands of interest:
        (20,60), (60,250), (250,500), (500,1000), (1000,2000), (2000,4000), (4000,6000)
        
        The output will be a list of tuples where each tuple contains the min and max index locations for each band
    '''
    def get_freq_loc(self):
        frequencies = np.arange(0,1 + self.n_fft/2) * self.sampling_rate/self.n_fft 
        cutoffs = [(20,60), (60,250), (250,500), (500,1000), (1000,2000), (2000,4000),(4000,6000)]
        locations = []
        
        for low, high in cutoffs:
            all_locs = np.where((frequencies >= low) & (frequencies <= high))[0]
            min_loc, max_loc = all_locs[0], all_locs[-1]
            locations.append((min_loc, max_loc))
        
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
        
        for bands in locs:
            min_l, max_l = bands
        
            energy = self.sftf[min_l:max_l+1, :]
            energy_db = librosa.amplitude_to_db(energy, ref=np.max)

            max_db = np.max(energy_db, axis=0)
            max_locs = np.argmax(energy_db, axis=0)
            
            peak_loc, _ = find_peaks(max_db, distance=43, prominence=20)
            
            for l in peak_loc:
                events.append((min_l + max_locs[l], l))
                 
        return events
    
    '''
        Function to use the events extracted to build a fingerprint of the song
        
        Will set an anchor point (b1, t1) and then look for other points (b2, t2) that occur after
        Will only store the difference between the anchor and target if is less than or equal to some maximum time delta
        default is set at 215 time steps or ~5 seconds
        
        The final output will be a list of tuples that stores the hash: anchor, target, and the time delta, and the time stamp
        ex. ((b1, b2, dt), t1) -> (hash, time stamp)
    '''
    def build_fingerprint(self, max_dt=215):
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
