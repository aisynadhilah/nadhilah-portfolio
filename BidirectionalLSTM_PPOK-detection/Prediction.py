import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")

def predict_all_folds(mfcc_array, model_dir, n_folds=5, threshold=0.61):
    """Run prediction on all fold models."""
    results = []

    for fold in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f"Model_Fold{fold}.keras")
        model = tf.keras.models.load_model(model_path, compile=False)

        t_steps, n_feat = model.input_shape[1], model.input_shape[2]
        needed = t_steps * n_feat

        flat = mfcc_array.flatten()
        if len(flat) < needed:
            flat = np.pad(flat, (0, needed - len(flat)))
        else:
            flat = flat[:needed]

        X = flat.reshape(1, t_steps, n_feat).astype("float32")
        prob = float(model.predict(X, verbose=0)[0][0])
        label = "Normal" if prob < threshold else "PPOK"

        results.append((prob, label))
        tf.keras.backend.clear_session()

    return results

def run_batch_prediction(data_folder, model_folder, output_csv, threshold=0.61):
    """Loop through all .npy MFCC files and save predictions to CSV."""
    file_list = sorted([f for f in os.listdir(data_folder) if f.endswith(".npy")])
    all_results = []

    for idx, fname in enumerate(file_list, start=1):
        fpath = os.path.join(data_folder, fname)

        try:
            mfcc = np.load(fpath)
            fold_results = predict_all_folds(mfcc, model_folder, threshold=threshold)

            probs = [round(p, 4) for p, _ in fold_results]
            labels = [lbl for _, lbl in fold_results]

            all_results.append([idx, fname] + probs + labels)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    columns = (
        ["no", "filename"] +
        [f"prob_fold_{i}" for i in range(1, 6)] +
        [f"pred_fold_{i}" for i in range(1, 6)]
    )

    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(output_csv, index=False)
    return df

if __name__ == "__main__":
    # Paths
    data_folder = "path/to/MFCC_npy"
    model_folder = "path/to/model_folder"
    output_csv = "predict_results.csv"

    threshold = 0.70

    df = run_batch_prediction(data_folder, model_folder, output_csv, threshold)
    print(f"\nPrediction completed. Output saved to: {output_csv}\n")
    print(df)
