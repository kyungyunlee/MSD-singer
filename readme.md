# Million Song Dataset for singer information retrieval

This repo contains code and information for singing-voice-based [Million Song Dataset](http://millionsongdataset.com/). It can be used for singing voice or singer relevant tasks. 

### Tensorflow/Keras version 
```
keras==2.0.8
tensorflow-gpu==1.4.0
```


### How it is created
Singing voice detection is performed on the Million Song Dataset (MSD).
Each singer has at least 20 songs with sufficient singing sections.    
For each singer, 20 songs are split into 15/2/3 for train/valid/test data.  



### How to use 
Under the `data` directory, `trainable_singers.txt` contains the list of singers in the MSD-singer dataset (total 5882 singers) and pickle files contain their track and vocal segments.  
```
import pickle 

# list of all the singers that can be used for training 
total_singers = [artist_id.strip('\n') for artist_id in open('data/trainable_singers.txt', 'r').readlines()]

# load the vocal segment data 
train_vocal_map = pickle.load(open('data/msd_artist_train_SVD.pkl', 'rb'))
valid_vocal_map = pickle.load(open('data/msd_artist_valid_SVD.pkl', 'rb'))
test_vocal_map = pickle.load(open('data/msd_artist_test_SVD.pkl', 'rb'))

# get train/valid/test tracks for a single singer 
curr_singer_id = total_singers[0]
print ('curr singer id:', curr_singer_id) 
train_tracks = train_vocal_map[curr_singer_id] 
print ('train track list:', list(train_tracks.keys()))
valid_tracks = valid_vocal_map[curr_singer_id]
test_tracks = test_vocal_map[curr_singer_id]
''' 
curr singer id: AR6681Y1187FB29B02
train track list: ['TRPOJCK128F92FDCD9', 'TRYCIWL128F932EC24', ...]
'''

# get a list of vocal segments for a single track of this singer 
curr_track = list(train_tracks.keys())[0]
print ('curr track id:', curr_track)
curr_track_vocal_segments = train_tracks[curr_track]
print ('vocal segments (sec):', curr_track_vocal_segments)
'''
curr track id: TRPOJCK128F92FDCD9
vocal segments (sec): [1.5, 6.0, 7.5, 9.0, 18.0, 19.5, 21.0, 30.0, 31.5, 40.5, 42.0, 43.5, 51.0, 52.5] 
'''
```




### Code used to compute SVD
SVD code and model is from [https://github.com/keums/SingingVoiceDetection](https://github.com/keums/SingingVoiceDetection).
If you want to use another SVD model, fix the model loading and configuration code accordingly.   
If you want to just run the code again, run the following with index from 1 to 5. (If code stops in the middle of the file, you need to start again. Increase the number of `TOTAL_INDEX` for smaller batches - default=5)      
```
cd svd_code  
python run_svd.py --index 1
```

### Project using this dataset 
* "Learning a Joint Embedding Space of Monophonic and Mixed Music Signals" Kyungyun Lee, Juhan Nam, ISMIR 2019 [https://github.com/kyungyunlee/mono2mixed-singer](https://github.com/kyungyunlee/mono2mixed-singer) 


