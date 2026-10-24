"""Constants shared across the pipeline.

Values follow "Integrating Hybrid Sampling with Machine Learning for Enhanced
Cardiovascular Disease Prediction" (Daniyal et al.).
"""

TARGET = "HeartDiseaseorAttack"

# Section 3.3: features most correlated with the target.
SELECTED_FEATURES = [
    "GenHlth",
    "Age",
    "DiffWalk",
    "HighBP",
    "Stroke",
    "PhysHlth",
    "Diabetes",
    "HighChol",
    "Smoker",
]

RANDOM_STATE = 42

# Section 3.4: 3,000 instances per class are held out as the test set.
TEST_PER_CLASS = 3000

# Section 3.4: 80% training / 20% validation.
VALIDATION_SIZE = 0.2

# Sampling techniques compared in Tables 3 and 4.
SAMPLING_METHODS = ["smoteenn", "ros", "hs"]
SAMPLING_LABELS = {"smoteenn": "SMOTEENN", "ros": "ROS", "hs": "HS"}

DL_MODELS = ["CNN", "NN", "MLP"]
ML_MODELS = ["DT", "SVM", "XGBoost"]

# Training settings for the DL models (Figs. 13-16: MLP 20 epochs, NN 10).
AUTOENCODER_DIM = 64
AUTOENCODER_EPOCHS = 10
DL_EPOCHS = {"CNN": 20, "NN": 10, "MLP": 20}
CV_EPOCHS = 10
BATCH_SIZE = 32

N_FOLDS = 5
