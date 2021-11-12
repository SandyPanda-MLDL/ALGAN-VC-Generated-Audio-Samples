# An Adaptive Learning based  Generative Adversarial Network for One-To-One Voice Conversion
***Sandipan Dhar, Student  Member,  IEEE, Nanda Dulal Jana, Member, IEEE, and Swagatam Das, Senior Member, IEEE.***
1. ArXiv link: https://arxiv.org/abs/2104.12159#

# Drive links of Converted Speech Samples
***Generated Samples by ALGAN-VC, ALGAN-VC without BLRS, ALGAN-VC without DRN, ALGAN-VC without L1 loss,  ALGAN-VC without L2 loss, CycleGAN-VC, CycleGAN-VC2, SP-CycleGAN***
1. ***All the speech samples generated or converted by the models are availabe at this Google drive link: https://drive.google.com/drive/folders/1fL-gpuDbkmNZgNMIIYCLA2BdYSiAxBvo?usp=sharing***
2. ***Some of the generated speech samples are presented at Google site ( link: https://sites.google.com/phd.nitdgp.ac.in/algan-vc-model/home )***
3. ***VCC 2016 speech dataset link: https://datashare.ed.ac.uk/handle/10283/2211***
4. ***VCC 2018 speech dataset link: https://datashare.ed.ac.uk/handle/10283/3061***
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
- ***ALGAN-VC-code Files with training and validation data*** 
  - ***1.main.py*** 
  - ***2.train.py*** 
  - ***3.model.py*** 
  - ***4.algan_network.py (based on Residual Network based connection)*** 
  - ***5.loss.py*** 
  
   ***(most importantly the Dense Residual Network based modified code is written in algan_dense_residual_net.py)***
   
  ***6.algan_dense_residual_net.py (use instead of algan_network.py to run with Dense Residual Network based connection)***
  - ***training_data of two speakers*** 
  - ***validation_data of two speakers***
  
******
- ***Optional arguments in main.py*** 
  - '--speaker_1_dir', type = str, help = 'Audio directory of the first speaker'
  - '--speaker_2_dir', type = str, help = 'Audio directory of the second speaker'
  - '--speaker_1_validation_dir', type = str, help = 'vocal style conversion of speaker_1  during  training.'
  - '--speaker_2_validation_dir', type = str, help = 'vocal style conversion of speaker_2  during  training.'
  - '--generated_data_directory', type = str, help = 'output directory for saving the generated audio samples'
  - '--epoch_number', type = int, help = 'epoch_number'
  - '--tfgan_model_directory', type = str, help = 'tfgan_model_directory for saving the model.'
  - '--tfgan_model_name', type = str, help = 'tfgan_model_name'
  - '--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.'
  - '--random_seed', type = int, help = 'random seed value'
  - '--batch_size', type = int, help = 'mini_batch_size'
  - '--g_learning_rate', type = int, help = 'generator_learning_rate'
  - '--d_learning_rate', type = int, help = 'discriminator_learning_rate'
  - '--sampling_rate', type = int, help = 'sampling_rate'
  - '--fp', type = int, help = 'frame period'
  - '--fn', type = int, help = 'frame number'
  
  ******

 - The ALGAN-VC code is developed based on the given repository code: https://github.com/leimao/Voice_Converter_CycleGAN
 - The objective evaluation codes are based on the given repository: https://github.com/r9y9/nnmnkwii

******

 # Objective evaluation codes (MCD, MSD and F0 RMSE) 
 - ***Jupyter Notebooks of calculating the Mel Cepstral Distortion (MCD) , F0 root means squared error (RMSE) and Modulation Spectra Distance (MSD) are available in this repository***
1. Code for Mel Cepstral Distortion (MCD).ipynb
2. F0 RMSE calculation.ipynb
3. MSD code implementation.ipynb


# Global Variance (GV), MCEP Trajectories and Modulation Spectra (MS) per modulation frequency

 - ***Jupyter Notebooks of calculating Global Variance (GV), MCEP Trajectories and Modulation Spectra (MS) per modulation frequency are available in this repository***
1. GV code.ipynb
2. MCEP_Trajectory.ipynb
3. MODULATION_SPECTRUM_CODE .ipynb
