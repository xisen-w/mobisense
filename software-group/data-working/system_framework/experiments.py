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
    

class IMU_Experiment_Setup:

    def __init__(self, experiment_name: str, experiment_description: str, experiment_data: dict, participant: Participant):
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.experiment_data = experiment_data
        self.participant = participant

    def run_experiment(self):
        pass
