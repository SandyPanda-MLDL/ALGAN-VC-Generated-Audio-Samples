# TFGAN-Generated-Audio-Samples
- The folder named ***Generated Samples by Proposed TFGAN*** contains the following  contents:
1. The audio samples used for training are in English for both the speakers and thus the generated samples also. (One Male speaker and one Female speaker)
2. The audio samples of the male speaker is in Bengali (An Indian regional language) for the similar experiment and thus one of the generated samples also.
- The folder named ***Self prepared dataset*** contains audio data of 10 different speakers from India -
# Our self prepared dataset contains Audio samples of 10 speakers
- ***Speaker1*** to ***Speaker10*** are ***10 different speakers*** and the recorded audios are in ***English Language***
  - ***The folder name (e.g Speaker_Number_gender_Region)*** represents the speaker number with their respective genders and the region they belong.
- ***SpeakerR1*** to ***SpeakerR5*** are ***5 different speakers*** and the recorded audios are in ***Five different Indian Regional Languages***
  - ***The folder name (e.g R_Speaker Number_gender name_Region the speaker belong)*** represents the speaker number with their respective genders and the regional language in which the audios were recorded.

***The linux command for converting the .mp4 .aac .mp3 etc audio file formats to .wav format is :***

- **for f in *.aac; do avconv -i "$f" "${f/%aac/wav}"; done*** replace the input format into the respective audio file formats accordingly 
# TFGAN-code 
***Experimental details  :***
The experiment was done in Python 3.6.9 and packages used
here for building the TF-GAN model are Tensorflow 1.15.0
and Keras 2.3.1. For audio data preprocessing Librosa 0.7.2
and Pyworld 0.2.8 version was used. For storing the feature
information in .npz format, Numpy 1.15 was used.
- ***TFGAN-code Files with training and validation data*** 
  - ***1.main.py*** 
  - ***2.train.py*** 
  - ***3.model.py*** 
  - ***4.tfgan_network.py*** 
  - ***5.loss.py*** 
  - ***training_data of two speakers*** 
  - ***validation_data of same two speakers***
  
