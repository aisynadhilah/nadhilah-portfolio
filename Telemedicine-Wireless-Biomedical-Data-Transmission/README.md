# Telemedicine Wireless Biomedical Signal Transmission

Proyek ini merupakan implementasi sistem telemedicine sederhana untuk mengambil, mengirim, menyimpan, dan menampilkan sinyal biomedis secara nirkabel menggunakan modul ESP8266, protokol TCP Socket, dan server berbasis C. Dalam demonstrasi, sinyal biomedis disimulasikan menggunakan gelombang sinus dari osilator, kemudian dikirim secara real-time menuju komputer server untuk visualisasi.

## Project Objectives
Tujuan utama proyek ini:
1. Mengakuisisi sinyal (disimulasikan oleh sinyal sinus) melalui ADC pada ESP8266.
2. Mengirimkan data secara real-time via WiFi menggunakan protokol TCP.
3. Menerima dan menyimpan data di server menggunakan program C multithread.
4. Menampilkan sinyal untuk dianalisis menggunakan Python.
5. Menerapkan arsitektur sistem dengan task & thread yang tersinkronisasi.

## System Architecture
### Hardware Components
1. Sinus signal generator (osilator)
2. Instrumentation amplifier
3. ESP8266 (ADC + WiFi TCP client)
4. Access Point (BME-WIFI)
5. Laptop/PC sebagai TCP server

### System Flow
[Oscillator] → [Instrumentation Amplifier] → [ESP8266] → WiFi → [TCP Server] → [Plot/Analysis]

## Software Implementation
A. ESP8266 (Client – ESP-IDF + FreeRTOS)

Tasks implemented:
- adc_read_task
  - Sampling ADC setiap 10 ms (100 Hz)
  - Menyimpan sampel pada shared buffer
- tcp_client_task
  - Mengirim 100 sampel/detik ke server
  - Menggunakan semaphore untuk sinkronisasi buffer
  - GPIO toggle untuk pengukuran timing via oscilloscope

Key features:
- Real-time, low-latency data streaming
- Semaphore-based buffer protection
- Task batching untuk stabilitas sistem

B. Server (C – Linux Environment)

Threads implemented:
- server_task
  - Membuka TCP socket
  - Menerima stream data
  - Menyimpan sementara di buffer bersama (mutex-protected)
- write_task
  - Menulis data ke file .txt secara teratur
  - Membersihkan format data (, menjadi newline)

Key functionalities:
- Multithreading (pthread)
- Mutex + Semaphore untuk sinkronisasi
- File handling stabil tanpa data loss

## Data Visualization

Data yang diterima server disimpan dalam file .txt.
Plot dilakukan menggunakan Python (Plotly) untuk menampilkan:
- Kurva sinyal sinus hasil pembacaan ADC
- Waktu vs amplitudo
- Keberlanjutan sinyal untuk validasi real-time streaming

## Performance Results
| Parameter                    | Hasil         |
| ---------------------------- | ------------- |
| **Sampling Rate**            | 100 Hz        |
| **ADC Read Time**            | ~1.78 ms      |
| **Time to Send 100 Samples** | ~5.81 ms      |
| **Data Loss**                | 0% (verified) |   

Karena waktu pengiriman (5.81 ms) < waktu sampling (10 ms), sistem berjalan stabil tanpa kehilangan data.

## Project Outcomes
1. Berhasil mengirim sinyal secara nirkabel dari ESP8266 ke server.
2. Proses penerimaan data berlangsung real-time dan stabil.
3. Sinyal berhasil divisualisasikan kembali untuk analisis.

## Key Skills Demonstrated
- Embedded system programming (ESP-IDF, FreeRTOS)
- Real-time ADC sampling
- TCP socket programming (client–server architecture)
- Multithreading (pthread) & synchronization primitives
- Data acquisition & biomedical signal simulation
- Python-based plotting & analysis
- Team coordination & technical reporting

## Author
Rihhadatul Aisy Nadhilah
- LinkedIn: https://www.linkedin.com/in/rihhadatulaisynadhilah/
- Email: ranadhilah17@gmail.com
