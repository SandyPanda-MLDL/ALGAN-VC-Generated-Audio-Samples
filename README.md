# An Adaptive Learning based  Generative Adversarial Network for One-To-One Voice Conversion
***Sandipan Dhar, Student  Member,  IEEE, Nanda Dulal Jana, Member, IEEE, and Swagatam Das, Senior Member, IEEE.***




1. ***Some of the generated speech samples are presented at Google site ( link: https://sites.google.com/phd.nitdgp.ac.in/algan-vc-model/home )***
2. ***VCC 2016 speech dataset link: https://datashare.ed.ac.uk/handle/10283/2211***
3. ***VCC 2018 speech dataset link: https://datashare.ed.ac.uk/handle/10283/3061***
4. ***VCC 2020 speech dataset link: https://github.com/nii-yamagishilab/VCC2020-database***
5. ***Emotional speech dataset link: https://github.com/HLTSingapore/Emotional-Speech-Data***
******
- The folder named ***Self prepared dataset*** contains audio data of 15 different speakers from India -
# Our self prepared dataset contains Audio samples of 15 speakers
- ***Speaker1*** to ***Speaker10*** are ***10 different speakers*** and the recorded audios are in ***English Language***
  - ***The folder name (e.g Speaker_Number_gender_Region)*** represents the speaker number with their respective genders and the region they belong.
- ***SpeakerR1*** to ***SpeakerR5*** are ***5 different speakers*** and the recorded audios are in ***Five different Indian Regional Languages***
  - ***The folder name (e.g R_Speaker Number_gender name_Region the speaker belong)*** represents the speaker number with their respective genders and the regional language in which the audios were recorded.

***The linux command for converting the .mp4 .aac .mp3 etc audio file formats to .wav format is :***

- **for f in *.aac; do avconv -i "$f" "${f/%aac/wav}"; done*** replace the input format into the respective audio file formats accordingly 

******
# ALGAN-VC-code 
***Experimental details  :***
The experiment was done in ***Python 3.6.9*** and packages used
here for building the ***ALGAN-VC*** model are ***Tensorflow 1.15.0***. For audio data preprocessing ***Librosa 0.7.2***
and ***Pyworld 0.2.8*** version was used. For storing the feature
information in ***.npz*** format, ***Numpy 1.15*** was used.

 - The ALGAN-VC code is developed based on the given repository code: https://github.com/leimao/Voice_Converter_CycleGAN
 - The objective evaluation codes are based on the given repository: https://github.com/r9y9/nnmnkwii

******

 # Objective evaluation codes (MCD, MSD and F0 RMSE) 
 - ***Jupyter Notebooks of calculating the Mel Cepstral Distortion (MCD) , F0 root means squared error (RMSE) and Modulation Spectra Distance (MSD) are available in this repository***
1. Code for Mel Cepstral Distortion (MCD).ipynb
2. F0 RMSE calculation.ipynb
3. MSD code implementation.ipynb


# Global Variance (GV), MCEP Trajectories , Modulation Spectra (MS) per modulation frequency and Perceptual Evaluation of Speech Quality (PESQ) 

 - ***Jupyter Notebooks of calculating Global Variance (GV), MCEP Trajectories and Modulation Spectra (MS) per modulation frequency and Perceptual Evaluation of Speech Quality (PESQ) are available in this repository***
1. GV code.ipynb
2. MCEP_Trajectory.ipynb
3. MODULATION_SPECTRUM_CODE .ipynb
4. PESQ code.ipynb
