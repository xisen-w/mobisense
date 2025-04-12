from typing import Dict, Any, Optional
import pandas as pd

class Participant:
    """Height, weight, age, gender, etc."""
    def __init__(self, participant_id, height, weight, age, gender, 
                stride_length=None, stride_number_per_minute=None,
                injury_day=None, injury_type=None):
        self.participant_id = participant_id
        self.height = height
        self.weight = weight
        self.age = age
        self.gender = gender
        self.stride_length = stride_length
        self.stride_number_per_minute = stride_number_per_minute
        self.injury_day = injury_day
        self.injury_type = injury_type

    def get_participant_data(self):
        return {
            "participant_id": self.participant_id,
            "height": self.height,
            "weight": self.weight,
            "age": self.age,
            "gender": self.gender,
            "stride_length": self.stride_length,
            "stride_number_per_minute": self.stride_number_per_minute,
            "injury_day": self.injury_day,
            "injury_type": self.injury_type
        }
    
    def build_participant_from_data(self, participant_data: dict):
        return Participant(
            participant_data["participant_id"], 
            participant_data["height"], 
            participant_data["weight"], 
            participant_data["age"], 
            participant_data["gender"], 
            participant_data.get("stride_length"), 
            participant_data.get("stride_number_per_minute"),
            participant_data.get("injury_day"),
            participant_data.get("injury_type")
        )

class Ankle_Sprained_Participant(Participant):
    def __init__(self, participant_id, height, weight, age, gender, 
                stride_length=None, stride_number_per_minute=None, 
                injury_day=None, injury_type=None, ankle_sprain_status=None):
        super().__init__(participant_id, height, weight, age, gender, 
                        stride_length, stride_number_per_minute, 
                        injury_day, injury_type)
        self.ankle_sprain_status = ankle_sprain_status

    def get_participant_data(self):
        data = super().get_participant_data()
        data["ankle_sprain_status"] = self.ankle_sprain_status
        return data
    

class IMU_Experiment_Setup:

    def __init__(self, experiment_name: str, experiment_description: str, experiment_data_path: str, participant: Participant):
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_data_path = experiment_data_path
        self.participant = participant
        self.experiment_data = None
        self.results_path = None
        self.has_angle_data = False

    def run_experiment(self):
        pass

    def load_experiment_data(self, filepath: str = None) -> pd.DataFrame:
        """Load experiment data from CSV file"""
        try:
            filepath = filepath or self.experiment_data_path
            
            # First check raw file for dorsiflexion_angle
            raw_angle_detected = False
            with open(filepath, 'r') as f:
                header = f.readline().strip()
                if 'dorsiflexion_angle' in header:
                    raw_angle_detected = True
                    print(f"Dorsiflexion angle found in raw CSV header of {filepath}")
                    # Get position of angle in header
                    header_parts = header.split(',')
                    angle_index = header_parts.index('dorsiflexion_angle')
                    # Read first line to check angle value
                    first_line = f.readline().strip()
                    if first_line:
                        data_parts = first_line.split(',')
                        if angle_index < len(data_parts):
                            angle_value = data_parts[angle_index]
                            print(f"First dorsiflexion angle value: {angle_value}")
            
            # Try standard loading first
            data = pd.read_csv(filepath)
            
            # Check if angle data is available
            self.has_angle_data = 'dorsiflexion_angle' in data.columns
            
            # If angle was in raw file but not in loaded data, try alternatives
            if raw_angle_detected and not self.has_angle_data:
                print(f"WARNING: Angle data found in raw CSV but not in pandas DataFrame for {filepath}")
                print("Attempting to fix with different loading options...")
                # Try with low_memory=False
                data = pd.read_csv(filepath, low_memory=False)
                self.has_angle_data = 'dorsiflexion_angle' in data.columns
                if self.has_angle_data:
                    print("Successfully recovered dorsiflexion angle data!")
            
            # Store the data
            self.experiment_data = data
            
            # Debug info
            if self.has_angle_data:
                print(f"Angle data detected in {filepath}")
                # Verify first few values
                print(f"First 5 angle values: {data['dorsiflexion_angle'].head().tolist()}")
            else:
                print(f"No angle data found in {filepath}")
                print(f"Available columns: {data.columns.tolist()}")
            
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    def get_experiment_info(self) -> Dict[str, Any]:
        """Return experiment information as a dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_description': self.experiment_description,
            'participant': self.participant.get_participant_data(),
            'data_path': self.experiment_data_path,
            'results_path': self.results_path,
            'has_angle_data': self.has_angle_data
        }
    
    def validate_experiment_data(self) -> bool:
        """Validate experiment data structure and content"""
        if self.experiment_data is None:
            self.load_experiment_data()
            
        if self.experiment_data is None:
            return False
            
        required_columns = ['imu0_timestamp', 'imu0_acc_x', 'imu0_acc_y', 'imu0_acc_z',
                          'imu0_gyro_x', 'imu0_gyro_y', 'imu0_gyro_z']
        
        # Check if all required columns exist in experiment data
        return all(col in self.experiment_data.columns for col in required_columns)

    def set_results_path(self, path: str):
        """Set the path for experiment results"""
        self.results_path = path

    def get_angle_data(self):
        """Get angle data if available"""
        if self.experiment_data is None:
            self.load_experiment_data()
            
        if not self.has_angle_data:
            return None
            
        return self.experiment_data['dorsiflexion_angle'] if 'dorsiflexion_angle' in self.experiment_data.columns else None

    
 