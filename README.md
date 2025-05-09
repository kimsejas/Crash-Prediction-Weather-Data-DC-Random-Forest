# Crash Prediction Using Weather Data

This project uses a Random Forest model to analyze how weather data impacts traffic crash prediction accuracy in Washington, DC (2014-2024). It evaluates the influence of specific weather conditions like temperature, precipitation, and wind speed.

## How It Works

1. **Data Preprocessing**:
- Merges crash and hourly weather data by timestamp.
- Adds features like hour of the day and day of the week.

2. **Model Evaluation**:
- Trains models with different feature sets:
    - Without weather data.
    - With all weather data.
    - With individual weather features (temperature, precipitation, wind speed).

3. **Results**:
- Outputs accuracy, confusion matrices, and classification reports.

## Usage 

### Cloning the Repository

This project uses **Git LFS** (Large File Storage) to manage large data files. To clone the repository, ensure you have Git LFS installed:

1. Install Git LFS (if not already installed):
   ```
   git lfs install
   ```

2. Clone the repository:
   ```
   git clone https://github.com/kimsejas/Crash-Prediction-Weather-Data-DC-Random-Forest.git
   ```

3. Pull the large files managed by Git LFS:
   ```
   git lfs pull
   ```

### Installing Dependencies

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the script: 
    ``` 
    python main.py
    ```
