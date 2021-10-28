# ALGAN-VC-Converted_Speech_Samples_Self_prepared_datasets (Generated Samples by Proposed ALGAN-VC and Converted_Speeches Folder contains the generated output samples 
)
- The folder named ***Converted_Speech_Samples_Self_prepared_dataset*** contains the following  contents:
1. The audio samples generated or converted during both intra gender and inter gender English language voice conversion. ( Male speaker and  Female speakers)
2. The audio samples generated or converted during both intra gender and inter gender Regional language voice conversion. ( Male speaker and  Female speakers)
3. The considered models are: ***ALGAN-VC, ALGAN-VC without BLRS, ALGAN-VC without DRN, ALGAN-VC without L1 loss,  ALGAN-VC without L2 loss, CycleGAN-VC, CycleGAN-VC2, SP-CycleGAN***
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
  - 
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

<script>
var audio = new Audio("https://raw.githubusercontent.com/USERNAME/REPOSITORY/BRANCH/Audio.mp3")
audio.play()
</script>
    
