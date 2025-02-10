from typing import Dict, Any
import pandas as pd

class Participant:
    """Height, weight, age, gender, etc."""
    def __init__(self, participant_id, height, weight, age, gender, stride_length, stride_number_per_minute):
        self.participant_id = participant_id
        self.height = height
        self.weight = weight
        self.age = age
        self.gender = gender
        self.stride_length = stride_length
        self.stride_number_per_minute = stride_number_per_minute

    def get_participant_data(self):
        return {
            "participant_id": self.participant_id,
            "height": self.height,
            "weight": self.weight,
            "age": self.age,
            "gender": self.gender,
            "stride_length": self.stride_length,
            "stride_number_per_minute": self.stride_number_per_minute
        }
    
    def build_participant_from_data(self, participant_data: dict):
        return Participant(participant_data["participant_id"], participant_data["height"], participant_data["weight"], participant_data["age"], participant_data["gender"], participant_data["stride_length"], participant_data["stride_number_per_minute"])

class Ankle_Sprained_Participant(Participant):
    def __init__(self, participant_id, height, weight, age, gender, stride_length, stride_number_per_minute, ankle_sprain_status):
        super().__init__(participant_id, height, weight, age, gender, stride_length, stride_number_per_minute)
        self.ankle_sprain_status = ankle_sprain_status

    def get_participant_data(self):
        pass 
    

class IMU_Experiment_Setup:

    def __init__(self, experiment_name: str, experiment_description: str, experiment_data_path: str, participant: Participant):
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_data_path = experiment_data_path
        self.participant = participant

    def run_experiment(self):
        pass

    def load_experiment_data(self, filepath: str) -> pd.DataFrame:
        """Load experiment data from CSV file"""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
        
    def get_experiment_info(self) -> Dict[str, Any]:
        """Return experiment information as a dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_description': self.experiment_description,
            'participant': self.participant.get_participant_data(),
            'data_path': self.experiment_data_path,
            'results_path': self.results_path
        }
    
    def validate_experiment_data(self) -> bool:
        """Validate experiment data structure and content"""
        required_columns = ['imu0_timestamp', 'imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z',
                          'imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z']
        
        # Check if all required columns exist in experiment data
        return all(col in self.experiment_data.keys() for col in required_columns)

    
 