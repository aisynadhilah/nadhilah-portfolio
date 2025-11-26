# akuisisi.py
import serial
import time
import os

PORT = '/dev/ttyACM0'      
BAUDRATE = 921600            
DURATION_SEC = 20
SAMPLING_RATE = 11111
TOTAL_SAMPLES = DURATION_SEC * SAMPLING_RATE

def akuisisi_data(filename: str) -> str:
    data = []
    buffer = bytearray()
    start_time = time.time()

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0)
        print(f"Koneksi berhasil ke {PORT} @ {BAUDRATE} baud.")
    except serial.SerialException as e:
        raise RuntimeError(f"Gagal membuka port {PORT}: {e}")

    try:
        while len(data) < TOTAL_SAMPLES and (time.time() - start_time) < DURATION_SEC:
            n = ser.in_waiting
            if n > 0:
                chunk = ser.read(n)
                buffer.extend(chunk)

                while len(buffer) >= 2:
                    msb = buffer[0]
                    lsb = buffer[1]
                    value = (msb << 8) | lsb
                    data.append(value)
                    del buffer[0:2]

        print(f"\nJumlah data terkumpul: {len(data)}")
    finally:
        ser.close()

    if len(data) == 0:
        raise RuntimeError("Tidak ada data yang terekam.")

    folder_path = "/home/lala/Tugas Akhir/AkuisisiData/Data"
    os.makedirs(folder_path, exist_ok=True)
    
    if not filename.endswith(".txt"):
        filename += ".txt"

    full_path = os.path.join(folder_path, filename)
    with open(full_path, 'w') as f:
        for value in data:
            f.write(f"{value}\n")

    return full_path 
