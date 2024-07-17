from scipy.io import wavfile

# Specify the filename of the WAV file
filename = 'filename.wav'

# Read the WAV file
sample_rate, data = wavfile.read(filename)
