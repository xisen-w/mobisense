from experiments import Participant
from experiments import IMU_Experiment_Setup
import pandas as pd

class Preprocessor:
    def __init__(self, experiment_setup: IMU_Experiment_Setup): #TODO: Make this optional 
        self.experiment_setup = experiment_setup

    def get_available_data(self):
        participant_data = self.experiment_setup.participant.get_participant_data()
        data_path = self.experiment_setup.experiment_data_path

    # Below are the functions for denoising the data

    def roll_mean_filter(self, data: pd.DataFrame, window_size: int):
        return data.rolling(window=window_size).mean()

    def low_pass_filter(self, data: pd.DataFrame, cutoff_frequency: float):
        return data.low_pass(cutoff_frequency)
    
    def kalman_filter(self, data: pd.DataFrame):
        pass
    
    def wavelet_filter(self, data: pd.DataFrame):
        pass
    
    def wavelet_denoise(self, data: pd.DataFrame):
        pass

    

    


    

    







