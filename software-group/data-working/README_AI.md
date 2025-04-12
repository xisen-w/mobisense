### Data Working

#### Data Collection: IMU & OpenCap

IMU is 20Hz to 27Hz; OpenCap is 60Hz. 

* Algorithm of Sampling Boosting: Both augment to 100Hz. 

#### Data Processing

Various filters are compared and introduced. 

#### Data Analysis

Max, Mean, & Etc. 

#### Data Visualisation

Done. 

#### RoM Calculation 

CHeck the angles & RoM. 

The default orientation is the rest orientation, which is directly on top and straight up-facing on the foot. 

#### Step Length Calculation 

- Calculate the step length. 

#### Anamoly Detection

- Use ONLY the normal data \& the paper's threshold to detect the anamoly. 
- Compare the normal data & the anamoly data to validate. 


#### Forece prediction. 

- Train a LSTM-CNN model to predict the force. 

### Experiment 

- We run for the three times. And record all the data. 
- We have the 'normal' and 'anomaly' data. 
- We have the 'rest' and 'dynamic' data.





