import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

FS = 11111

def plot_signal(file_path, start_sec=0, duration=20, return_fig=False):
    with open(file_path, 'r') as f:
        data = [int(line.strip()) for line in f.readlines()]
        
    data = data - np.mean(data) #baseline

    total_len = len(data)
    start_idx = int(start_sec * FS)
    end_idx = start_idx + int(duration * FS)
    sliced_data = data[start_idx:end_idx]
    waktu = np.linspace(start_sec, start_sec + duration, len(sliced_data))

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(waktu, sliced_data)
    ax.set_title("Sinyal Suara Paru-Paru)")
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Amplitudo")
    ax.grid(True)
    fig.tight_layout()

    if return_fig:
        return fig
    else:
        os.makedirs("output", exist_ok=True)
        fig.savefig("output/plot_signal.png")
        plt.close(fig)
        return "output/plot_signal.png"
