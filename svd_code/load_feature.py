import os.path
import sys
import numpy as np
import librosa

def featureExtract(FILE_NAME):
    print (FILE_NAME)
    try:
        y, _ = librosa.load(FILE_NAME, sr=16000)
        # print ("duration", librosa.get_duration(y, sr=16000))
        mel_S = librosa.feature.melspectrogram(y, sr=16000, n_fft=1024, hop_length=160, n_mels=80)
        log_mel_S = librosa.power_to_db(mel_S,ref=np.max)
        log_mel_S = log_mel_S.astype(np.float32)
        return y, log_mel_S

    except Exception as ex:
        print('ERROR: ', ex)
        
def makingTensor(feature,stride):
    num_frames = feature.shape[1]
    x_data = np.zeros(shape=(num_frames, 75, 80, 1))
    total_num = 0
    HALF_WIN_LEN = 75  // 2

    for j in range(HALF_WIN_LEN, num_frames - HALF_WIN_LEN -2, stride): # was -2 but changed to -1 
        mf_spec = feature[:, range(j - HALF_WIN_LEN, j + HALF_WIN_LEN + 1)]
        x_data[total_num, :, :, 0] = mf_spec.T
        total_num = total_num + 1

    x_data = x_data[:total_num]

    x_train_mean = np.load('./x_data_mean_svad_75.npy')
    x_train_std = np.load('./x_data_std_svad_75.npy')
    x_test = (x_data - x_train_mean) / (x_train_std + 0.0001)
    return x_test


if __name__ == '__main__':
    featureExtract()
