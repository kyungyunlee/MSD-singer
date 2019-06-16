import os
import sys
import librosa
import numpy as np
import pickle
from scipy.signal import medfilt 
import argparse
from keras.optimizers import Adam

from model_SVAD import * 
from load_feature import * 
from config import HPARAMS

os.environ["CUDA_VISBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True)
args = parser.parse_args()


DATA_PATH = '/home1/irteam/users/jongpil/data/msd/songs/' 
msd_to_id7d = pickle.load(open('MSD_id_to_7D_id.pkl', 'rb'))
id7d_to_path = pickle.load(open('7D_id_to_path.pkl', 'rb'))
TOTAL_INDEX = 5 

# load model 
model = SVAD_CONV_MultiLayer()
weight_name = 'weights/SVAD_CNN_ML.hdf5'
opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.load_weights(weight_name)






def segment_file(y_pred):
    ''' 
    Finds the vocal segments. At least 70 percent of the total segment should be vocal. 
    '''
    idx = 0 
    total_seg = 0
    time_seg = [] 
    while (idx < len(y_pred) - HPARAMS.timeLength):
        sum_seg = sum(y_pred[idx:idx+HPARAMS.timeLength])
        if sum_seg >= HPARAMS.timeLength  * HPARAMS.singingPercent : 
            time_seg.append(idx * 0.01)
            total_seg +=1 

        idx = idx + int(HPARAMS.timeLength * 0.5)

    return time_seg

def detect_vocal_segments(filename):
    y, feature = featureExtract(filename)
    x_test = makingTensor(feature, stride=HPARAMS.stride)
    y_pred = model.predict(x_test, verbose=1)
    y_pred = medfilt(y_pred[:,0], HPARAMS.mf_window_size)
    y_pred = (y_pred > HPARAMS.threshold)
    y_pred = y_pred.astype('int8')

    # pad front and back 
    y_pred_pad = np.zeros(x_test.shape[0] + HPARAMS.win_len)
    y_pred_pad[HPARAMS.win_len //2: -HPARAMS.win_len//2] = y_pred

    if y_pred[0] == 1 :
        y_pred_pad[:HPARAMS.win_len //2 +1] = 1
    if y_pred[-1] == 1 :
        y_pred_pad[-HPARAMS.win_len//2 :] = 1
    
    # get continuous vocal segments
    seg_idx = segment_file(y_pred_pad)
    return seg_idx
            




def main():
    # place to save results
    train_songs_by_singers = {}
    test_songs_by_singers = {}
    trainable_singers = []

    # load data
    train_f = open('../data/MSD_train.txt').readlines()
    train_list = {} # key : singer, value : list of songs

    for x in train_f:
        l = x.split('<SEP>')
        song = l[0].strip()
        singer = l[1].strip('\n')
        try : 
            train_list[singer].append(song)
        except KeyError:
            train_list[singer] = []
            train_list[singer].append(song)

    test_f = open('../data/MSD_test.txt').readlines()
    test_list = {} 

    for x in test_f: 
        l = x.split('<SEP>')
        song = l[0].strip()
        singer = l[1].strip('\n')
        try : 
            test_list[singer].append(song)
        except KeyError:
            test_list[singer] = []
            test_list[singer].append(song)

    print (len(train_list), len(test_list))
    
    # Get a subset of the artists to work on 
    all_artists = list(train_list.keys())
    num_sub_artists = len(all_artists) // TOTAL_INDEX

    if args.index  == TOTAL_INDEX: 
        curr_artists = all_artists[(args.index -1)* num_sub_artists:]
    else: 
        curr_artists  = all_artists[(args.index-1)*num_sub_artists : args.index * num_sub_artists]

    print ("working on index {} with {} artists".format(args.index, len(curr_artists)))

    
    # run 
    num_artist_done = 0 
    num_valid_artist = 0

    for artist_id in curr_artists: 
        train_songs_by_singers[artist_id] = {}
        test_songs_by_singers[artist_id] = {}
        num_valid_train_songs = 0
        num_valid_test_songs = 0

        artist_train_songs = train_list[artist_id]
        artist_test_songs = test_list[artist_id]
        print ("num train songs {}, num test songs {}".format(len(artist_train_songs), len(artist_test_songs)))

        print ("Train songs for {}...".format(artist_id))
        for artist_song in artist_train_songs:
            if num_valid_train_songs >= 15:
                print ("artist {} has at least 15 valid  train songs, total {}".format(artist_id, num_valid_train_songs))
                break

            else :
                
                try : 
                    clippath = id7d_to_path[msd_to_id7d[artist_song]]
                    filename = os.path.join(DATA_PATH, clippath)
                    # get valid vocal segments 
                    seg_idx = detect_vocal_segments(filename)
                    print ("num vocal segments", len(seg_idx))

                    # if there are more than 10 segments
                    if len(seg_idx) >= 10:
                        train_songs_by_singers[artist_id][artist_song] = seg_idx 
                        num_valid_train_songs +=1 
                    # print ("num artist done {}/{}. valid {}".format(num_artist_done, len(curr_artists), len(trainable_singers)))

                except :
                   continue 
        print("Test songs for {}...".format(artist_id))
        for artist_song in artist_test_songs:
            if num_valid_test_songs >= 5 : 
                print ("artist {} has at least 5 valid test songs, total {}".format(artist_id, num_valid_test_songs))
                break

            else :
                try :
                    clippath = id7d_to_path[msd_to_7d[artist_song]]
                    filename = os.path.join(DATA_PATH, clippath)
                    # get valid vocal segments 
                    seg_idx = detect_vocal_segments(filename)
                    print ("num vocal segments", len(seg_idx))

                    if len(seg_idx) >= 10 :
                        train_songs_by_singers[artist_id][artist_song] = seg_idx

                except : 
                    continue
        
        # there should be 15 valid train songs and 5 valid test songs to be considered as a "singer"  
        if num_valid_train_songs == 15 and num_valid_test_songs == 5 : 
            trainable_singers.append(artist_id)
            num_valid_artist += 1 
            print ("trainable singer found")
        else : 
            print ("not enough songs for artist {} : {} train, {} test".format(artist_id, num_valid_train_songs, num_valid_test_songs))

        num_artist_done += 1 

        print ("{}/{} done. number of trainable singers {}".format(num_artist_done, len(curr_artists), len(trainable_singers)))

    
    # save the result 
    train_pickle_f = "msd_artist_train_SVD_" + str(args.index) + ".pkl"
    pickle.dump(train_songs_by_singers, open(train_pickle_f, 'wb'))
    
    # divide into valid/test 
    test_by_singers = {}
    valid_by_singers = {}
    for artist, songs in test_songs_by_singers.items():
        songs = list(songs.items())
        test_tmp = {x[0]:x[1] for x in songs[:3]}
        valid_tmp = {x[0]:x[1] for x in songs[3:]}
        test_by_singers[artist] = test_tmp
        valid_by_singers[artist] = valid_tmp 

    valid_pickle_f = "msd_artist_valid_SVD_" + str(args.index) + ".pkl"
    pickle.dump(valid_by_singers, open(valid_pickle_f, 'wb'))

    test_pickle_f = "msd_artist_test_SVD_" + str(args.index) + ".pkl"
    pickle.dump(test_by_singers, open(test_pickle_f, 'wb'))

    with open("trainable_singers_" + str(args.index) + ".txt", "w") as f:
        for singer in trainable_singers:
            f.write(singer +'\n')




if __name__ == '__main__':
    main()
