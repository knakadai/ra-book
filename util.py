# 各種パッケージの読み込み
import numpy as np
import sys
import os
import math
import struct
import re
import soundfile as sf              # soundfile パッケージ
import scipy                          # scipy
#import urllib
import urllib.request               # ダウンロード
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython.display              # 音声埋め込み用
import numpy.linalg as LA
import gdown

# 時間波形のプロット
def plotwave(wave, fs, xrange = -1):
    # 信号長
    total = wave.shape[0]/fs
    if(xrange == -1):
        xrange = total

    # 表示
    time = np.arange(0, total, 1/fs)

    # 表示
    plt.figure()
    plt.plot(time, wave)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim([0,xrange])
    plt.grid()

# スペクトルのプロット
def plotspectrum(wave, fs, yscale = "dB"):
    # 高速フーリエ変換
    spec = np.fft.fft(wave)

    # 軸ラベル作成
    if yscale == "dB": # log scale
        power2 = 20 * np.log10(np.abs(spec) + ε)
    else: # linear scale
        power2 = 20 * np.abs(spec)
    freq2 = np.fft.fftfreq(len(wave)) * fs /1000
    freq = freq2[:len(freq2)//2]
    power = power2[:len(power2)//2]

    # 表示
    plt.figure()
    plt.plot(freq, power)
    plt.xlabel('Frequency [kHz]')
    if yscale == "dB": # log scale
        plt.ylabel('Power [dB]')
    else: # linear scale
        plt.ylabel('Power')
    plt.grid()

# スペクトログラムのプロット
def plotspectrogram(wave, fs, flen=512, slen=160):
    # 信号長
    total = wave.shape[0]/fs

    # 時間軸
    time = np.arange(0, total, 1/fs)

    # 短時間フーリエ変換のためのフレーム化
    if np.ndim(wave) == 1:
        wave = wave.reshape(-1,1)
    frames = np.lib.stride_tricks.sliding_window_view(wave, flen, axis=0)[::slen,:,:]

    # 窓関数 ハミング窓
    win = np.hamming(flen+1)[:-1]
    data = frames * win

    # データの次元数の整形
    data = np.squeeze(data.T)

    # 高速フーリエ変換
    spectrogram = np.fft.fft(data, axis=0)[:(flen//2+1)]

    # 対数パワー
    spectrogram = 20 * np.log10(np.abs(spectrogram) + ε)

    # 表示
    f, ax = plt.subplots()
    ax.imshow(spectrogram, origin='lower', cmap='viridis', extent=(0, 1, 0, fs/2/1000))
    ax.axis('tight')
    ax.set_ylabel('Frequency[kHz]')
    ax.set_xlabel('Time[s]')


# 時間波形とスペクトログラムのプロット
def plotwave_and_spectrogram(wave, fs, xrange=-1):
    # プロット

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)

    total = wave.shape[0]/fs
    time = np.arange(0, total, 1/fs)
    if xrange == -1:
        dwave = wave
    else:
        time = time[0:xrange]
        dwave = wave[0:xrange]
    ax1.plot(time, dwave)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Amplitude')
    ax1.grid()

    ax2 = fig.add_subplot(2,2,2)
    _, _, _, _ = ax2.specgram(wave, NFFT=512, Fs=fs, noverlap=160)
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Frequency [Hz]')

    plt.tight_layout()

# google drive 上の音声ファイル読込みと表示
def readwavefile(id, fname):
    url = f'https://drive.google.com/uc?id={id}'
    # urllib.request.urlretrieve(url, fname)
    gdown.download(url, fname)

    # ファイル読込
    data, fs = sf.read(fname)

    # プロット
    # plotwave(data,fs)

    #埋め込み
    # IPython.display.display(IPython.display.Audio(data,rate=fs))

    return data, fs

# インパルス応答の可視化（サンプル数512, 16kHz, 多チャンネル)
def showimp3d(imp):
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    len = 512
    x = np.array(range(len))/16 #16kHz, [ms]
    xlabel = "ms"

    for i in range(8):
        y = np.full(len,i)
        z = imp[0:len,i]
        ax.plot(x, y, z)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("microphone")


# 伝達関数の可視化
# dir : 音源方向インデックス
# mode : time/freq
def showtf3d(tfdata, dir = 0, mode="time"):
    if mode != "time":
        mode = "freq"
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    if mode == "time":
        len = 50
        x = np.array(range(len))/16 #16kHz, [ms]
        xlabel = "ms"
    elif mode == "freq":
        len = 257
        x = np.array(range(len))*8/256 #16kHz, [kHz]
        xlabel = "kHz"

    for i in range(8):
        y = np.full(len,i)
        if mode == "time":
            z = np.real(np.fft.ifft(np.squeeze(tfdata[i,0:50,dir])))
        elif mode == "freq":
            z = np.abs(np.squeeze(tfdata[i,:,dir]))
        ax.plot(x, y, z)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("microphone")

# 特定の音源方向の伝達関数 (matファイル)を読み込む
def _read_tfmat(f):
    tag = f.read(32)
    tag = tag.decode().strip()
    dtype = f.read(32)
    dtype = dtype.decode().strip()

    dim = int.from_bytes(f.read(4), 'little')
    nrow = int.from_bytes(f.read(4), 'little') #first
    ncol = int.from_bytes(f.read(4), 'little') #second

    if dtype != 'complex':
        print('dtype error')
        return -1
    if dim != 2:
        print('dim error')
        return -1
    data = np.zeros((nrow, ncol), np.complex64)
    for j in range(nrow):
        for i in range(ncol):
            real = f.read(4)
            real = struct.unpack('<f', real)[0]
            imag = f.read(4)
            imag = struct.unpack('<f', imag)[0]
            data[j,i] = real + imag*1j
            #print(data[j,i])
    return data

# HARK の伝達関数サイズ取得 マイク数 * 周波数ビン数 * 音源方向数, 伝達関数名リスト
def _get_tfsize(zf, type = 'separation'):
    if type != "separation":
        type = 'localization'
    tflist = [x for x in zf.namelist() if re.match('transferFunction/'+type+'/tf[\d]+.mat', x)]
    ndir = len(tflist)
    f = zf.open(tflist[0])
    tag = f.read(32)
    tag = tag.decode().strip()
    dtype = f.read(32)
    dtype = dtype.decode().strip()

    dim = int.from_bytes(f.read(4), 'little')
    nrow = int.from_bytes(f.read(4), 'little') #first
    ncol = int.from_bytes(f.read(4), 'little') #second

    f.close()

    if dtype != 'complex':
        print('dtype error')
        return -1
    if dim != 2:
        print('dim error')
        return -1
    return nrow, ncol, ndir, tflist

# HARK の伝達関数読み込む
def readtffile(tffile, type = 'separation'):
    # 伝達関数ダウンロード
    if not os.path.isfile(tffile):
        print("Download and Load: "+tffile)
        if tffile == "tamago_rectf.zip":
            url = f'https://drive.google.com/uc?export=download&id=1FMUXjGY7NLiINOdag8feJb8NS6Aobynf'
        elif tffile == "tamago_geotf.zip":
            url = f'https://drive.google.com/uc?export=download&id=1zCPmb1wIZYMLtyURD7cKOa2W2-9Sje0_'
        elif tffile == "dacho_geotf_v3.zip":
            url = f'https://drive.google.com/uc?export=download&id=1-wJ0o8fR_T4tO1sD-jp_82v4RnYXcuJV'
        else:
            print("No File Existed: "+tffile)
            return
        urllib.request.urlretrieve(url, tffile)
    else:
        print("Load Already Existing File: "+tffile)

    zf = ZipFile(tffile)
    nrow, ncol, ndir, tflist = _get_tfsize(zf, type)
    data = np.zeros((nrow, ncol, ndir), np.complex64)
    for i, tfname in enumerate(sorted(tflist)):
        f = zf.open(tfname, 'r')
        data[:,:,i] = _read_tfmat(f)
        f.close()
    zf.close()
    # HARK では，分離用の伝達関数は共役転置がかかっているので戻す。
    if type == 'separation':
        data = data.conjugate()
    print("#mics {0:d}, #freqs {1:d}, #dirs {2:d}".format(nrow, ncol, ndir))
    return data

# 物理定数等定義
π = math.pi # 円周率
v = 340      # 音速 340 [m/s]
ε = sys.float_info.epsilon
