import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

A = 10  # Set the limit for the x-axis
N = 1000  # Number of data points
x = np.linspace(-3*A, 3*A, N)  # Generate x-values within the range

step_signal_1d = np.where(np.abs(x) <= 1, 1, 0)

fft_result_1d = np.fft.fftshift(np.fft.fft(step_signal_1d))
freq_1d = np.fft.fftshift(np.fft.fftfreq(N, x[1] - x[0]))

step_signal_2d = np.outer(step_signal_1d, step_signal_1d)

fft_result_2d = np.fft.fftshift(np.fft.fft2(step_signal_2d))
freq_2d = np.fft.fftshift(np.fft.fftfreq(N, x[1] - x[0]))

def calculate_mfcc(signal, samplerate):
    nfft = 512
    mfcc_result = mfcc(signal, samplerate=samplerate, numcep=13, nfilt=26, winfunc=np.hamming, nfft=nfft)
    return mfcc_result

samplerate = N
mfccs = calculate_mfcc(step_signal_1d, samplerate)

plt.figure(figsize=(18, 9))

plt.subplot(2, 2, 1)
plt.step(x, step_signal_1d)
plt.title('1D Step Signal with a Peak of 1')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(freq_1d, np.abs(fft_result_1d))
plt.title('1D FFT of the 1D Step Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 2, 3)
plt.imshow(np.abs(fft_result_2d), extent=(freq_2d.min(), freq_2d.max(), freq_2d.min(), freq_2d.max()))
plt.title('2D FFT of the 2D Step Signal')
plt.xlabel('Frequency (X)')
plt.ylabel('Frequency (Y)')
plt.colorbar()
plt.grid()

plt.subplot(2, 2, 4)
plt.imshow(mfccs, cmap='viridis', origin='lower', aspect='auto')
plt.title('MFCC of the 1D Step Signal')
plt.xlabel('MFCC Coefficient')
plt.ylabel('Frame Index')
plt.colorbar()
plt.grid()

plt.tight_layout()
plt.show()