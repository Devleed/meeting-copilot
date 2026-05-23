import sounddevice as sd
import numpy as np

def test_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    volume = np.abs(indata).mean()
    print(f"Volume: {volume:.4f}")

try:
    with sd.InputStream(device=8, channels=1, samplerate=48000, callback=test_callback):
        input("Press Enter to stop...")
except Exception as e:
    print(f"Error: {e}")