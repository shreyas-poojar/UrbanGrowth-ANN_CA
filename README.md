🌆 Urban Growth Prediction using ANN + Cellular Automata

📌 Project Overview
This project predicts future urban expansion using an Artificial Neural Network (ANN) combined with Cellular Automata (CA). It uses time-series LULC (Land Use Land Cover) raster data to learn land transition patterns and simulate future city growth.

📍 Study Area: HUBLI
📅 Predictions:

*  10 years → [2015-2025]


🧭 Workflow
Raster Data Loading → Data Preprocessing → Feature Extraction → ANN Training → Transition Prediction → Cellular Automata → Future Map Generation → Visualization

📂 Project Structure
LULC-CA-Markov-Prediction/
│
├── data/                # Input raster files (2015, 2020, 2025)
├── output/              # Predicted rasters (2035, 2045)
├── scripts/             # Core implementation
│   ├── load_data.py
│   ├── transition_matrix.py
│   ├── markov_model.py
│   ├── cellular_automata.py
│   └── predict.py
│
├── requirements.txt
└── README.md

🚀 How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Train the ANN model
   python scripts/train_ann.py

3. Run prediction
   python scripts/predict.py

📊 Outputs

*predicted_urban_growth_2030.tif  → Future urban growth map
* Transition probability maps
* Visualization of urban expansion

🔥 Features
✔ ANN-based land transition learning
✔ Cellular Automata for spatial simulation
✔ Time-series LULC analysis
✔ GeoTIFF output support
✔ Works for different cities

📈 Future Improvements

* Accuracy assessment (Kappa coefficient)
* Improved CA neighborhood rules
* Integration with deep learning models
* Web-based visualization

👩‍💻 Author
SHREYAS S B
