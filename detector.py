import librosa
import numpy as np
import sounddevice as sd
from scipy.stats import skew, kurtosis

def analyze_voice(audio_data, sample_rate):
    # 1. Feature Extraction
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    
    # 2. Calculate Statistics
    # Ang AI voices ay madalas kulang sa "natural variance" o ingay ng tao
    mean_mfcc = np.mean(mfccs)
    std_mfcc = np.std(mfccs)
    voice_skew = skew(spectral_centroids)
    
    # Simple Logic Threshold (Sa totoong system, ito ay Machine Learning Model)
    # Kapag masyadong 'flat' o 'consistent' ang boses, suspect ito as Deepfake
    is_synthetic = False
    if std_mfcc < 15 or voice_skew < 1.0: 
        is_synthetic = True
        
    return is_synthetic, std_mfcc

def start_vishing_monitor():
    fs = 22050  # Sample rate
    seconds = 3  # Titingin kada 3 segundo
    
    print("🟢 System Active: Listening for suspicious audio patterns...")
    
    while True:
        # Recording live audio
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        
        # Flatten the array
        audio_flat = recording.flatten()
        
        # Analyze
        is_fake, score = analyze_voice(audio_flat, fs)
        
        if is_fake:
            print(f"⚠️ ALERT: Potential Deepfake/Vishing Detected! (Variance Score: {score:.2f})")
        else:
            print(f"✅ Audio Clear (Variance Score: {score:.2f})")

if __name__ == "__main__":
    start_vishing_monitor()