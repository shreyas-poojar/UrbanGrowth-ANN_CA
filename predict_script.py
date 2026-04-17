# ==============================
# 1. IMPORTS
# ==============================
import os
from pyexpat import model
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, distance_transform_edt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ==============================
# 2. USER INPUT
# ==============================
LULC_MAPS = [
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2015.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2016.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2017.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2018.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2019.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2020.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2021.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2022.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2023.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2024.tif",
    r"C:\Users\Phoenix Enterprises\OneDrive\Desktop\New folder\city _prediction_project\LULC_10m_DW_Admin_Hubli_2025.tif"
]

OUTPUT_NAME = "predicted_urban_growth_2030.tif"
URBAN_CLASS_ID = 3
CA_WINDOW_SIZE = 3
SIMULATION_STEPS = 5

# ==============================
# 3. CA FUNCTION
# ==============================
def get_ca_neighborhood(data, urban_val, size):
    urban_binary = (data == urban_val).astype(float)
    density = uniform_filter(urban_binary, size=size)
    return density

# ==============================
# 4. TRAIN MODEL
# ==============================
def train_model():
    X_train_list = []
    y_train_list = []

    print("Preparing training data...")

    for i in range(len(LULC_MAPS) - 1):
        print(f"  [{i+1}/{len(LULC_MAPS)-1}] Reading {os.path.basename(LULC_MAPS[i])}...", flush=True)

        with rasterio.open(LULC_MAPS[i]) as t1, rasterio.open(LULC_MAPS[i+1]) as t2:
            data1 = t1.read(1, out_shape=(t1.height // 2, t1.width // 2))
            data2 = t2.read(1, out_shape=(t2.height // 2, t2.width // 2))

            print(f"  [{i+1}/{len(LULC_MAPS)-1}] Computing CA neighborhood...", flush=True)
            nb = get_ca_neighborhood(data1, URBAN_CLASS_ID, CA_WINDOW_SIZE)

            print(f"  [{i+1}/{len(LULC_MAPS)-1}] Computing distance transform...", flush=True)
            dist = distance_transform_edt(data1 != URBAN_CLASS_ID)

            X = np.column_stack([data1.ravel(), nb.ravel(), dist.ravel()])
            y = (data2 == URBAN_CLASS_ID).astype(int).ravel()

            X_train_list.append(X)
            y_train_list.append(y)
            print(f"  [{i+1}/{len(LULC_MAPS)-1}] ✅ Done", flush=True)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    MAX_SAMPLES = 2_000_000
    if len(X_train) > MAX_SAMPLES:
        print(f"Subsampling {len(X_train):,} → {MAX_SAMPLES:,} rows...", flush=True)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), MAX_SAMPLES, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    print("Scaling features...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("Training ANN model...", flush=True)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, verbose=True)
    model.fit(X_scaled, y_train)

    print("✅ Model trained!", flush=True)
    return model, scaler

# ==============================
# 5. SIMULATION
# ==============================
def simulate_future(model, scaler, shape):
    with rasterio.open(LULC_MAPS[-1]) as src:
        current_map = src.read(1, out_shape=(src.height // 2, src.width // 2))
        meta = src.profile

    print("Running CA simulation...")

    for step in range(SIMULATION_STEPS):
        print(f"  Step {step+1}/{SIMULATION_STEPS}", flush=True)

        nb = get_ca_neighborhood(current_map, URBAN_CLASS_ID, CA_WINDOW_SIZE)
        dist = distance_transform_edt(current_map != URBAN_CLASS_ID)

        X = np.column_stack([current_map.ravel(), nb.ravel(), dist.ravel()])
        X_scaled = scaler.transform(X.astype(np.float32))

        # Predict in chunks to avoid memory error
        CHUNK = 100_000
        prob = np.empty(len(X_scaled), dtype=np.float32)
        for start in range(0, len(X_scaled), CHUNK):
            end = min(start + CHUNK, len(X_scaled))
            prob[start:end] = model.predict_proba(X_scaled[start:end])[:, 1]

        prob_map = prob.reshape(current_map.shape)
        new_map = current_map.copy()
        new_map[prob_map > 0.6] = URBAN_CLASS_ID
        current_map = new_map

    meta.update(dtype=rasterio.int16, count=1,
                width=current_map.shape[1], height=current_map.shape[0])

    with rasterio.open(OUTPUT_NAME, 'w', **meta) as dst:
        dst.write(current_map.astype(rasterio.int16), 1)

    print("\n✅ Simulation complete!")
    print(f"📁 Saved to: {OUTPUT_NAME}")
# ==============================
# 6. MAIN
# ==============================
def main():
    for f in LULC_MAPS:
        if not os.path.exists(f):
            print(f"❌ Missing file: {f}")
            return

    with rasterio.open(LULC_MAPS[0]) as src:
        shape = (src.height // 2, src.width // 2)

    model, scaler = train_model()
    simulate_future(model, scaler, shape)

if __name__ == "__main__":
    main()