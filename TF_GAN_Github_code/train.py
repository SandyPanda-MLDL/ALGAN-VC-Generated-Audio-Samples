import os
import numpy as np
import time
import librosa
import pyworld
from model import *

def librosa_load(speaker_dir, sr):

    time_series = list()
    for file in os.listdir(speaker_dir):
        file_path = os.path.join(speaker_dir, file)
        audio_time_series, s_rate = librosa.load(file_path, sr = sr, mono = True)
        
        time_series.append(audio_time_series)

    return time_series
def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded
#encode data as per WORLD analysis
def encode_data(time_series, sr , fp , mcep):

    f0_freq = list()
    timeaxes = list()
    spectrogram = list()
    aperiodicity = list()
    coded_sps = list()

    for data in time_series:
        f0, timeaxis, sp, ap = world_decompose(data, sr, fp)
        coded_sp = world_encode_spectral_envelop(sp, sr, mcep)
        f0_freq.append(f0)
        timeaxes.append(timeaxis)
        spectrogram.append(sp)
        aperiodicity.append(ap)
        coded_sps.append(coded_sp)

    return f0_freq, timeaxes, spectrogram,aperiodicity, coded_sps

#decompose speech signal into f0, spectral envelope and aperiodicity using WORLD analysis
def world_decompose(data, sr, fp):

    
    data =data.astype(np.float64)
    # Harvest F0 extraction algorithm.
    f0_freq, timeaxis = pyworld.harvest(data, sr, fp , f0_floor = 71.0, f0_ceil = 800.0)
    # CheapTrick harmonic spectral envelope estimation algorithm.
    spectrogram = pyworld.cheaptrick(data , f0_freq, timeaxis, sr)
    # D4C aperiodicity estimation algorithm.
    aperiodicity= pyworld.d4c(data, f0_freq, timeaxis, fp)

    return f0_freq, timeaxis, spectrogram, aperiodicity

#mel-cepstral coefficients (MCEPs)
def world_encode_spectral_envelop(sp, sr, mcep):

    coded_sps = pyworld.code_spectral_envelope(sp, sr, mcep)

    return coded_sps

def world_decode_spectral_envelop(coded_sp, sr):

    fftlen = pyworld.get_cheaptrick_fft_size(sr)
    
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, sr, fftlen)

    return decoded_sp

def world_speech_synthesis(f0, decoded_sp, ap, sr, fp):

    
    wav = pyworld.synthesize(f0, decoded_sp, ap, sr, fp)
    
    wav = wav.astype(np.float32)

    return wav

#mean and standard deviation of logarithmic fundamental frequency 
def logf0_statistics(f0s_speaker):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s_speaker))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

#transpose of coded_sps
def transpose_coded_sps(lst):

    transposed_lst = list()
    for elements in lst:
        transposed_lst.append(elements.T)
    return transposed_lst

#normalized coded sps
def normalized_coded_sps(coded_sps):

    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted
def sample_train_data(dataset_speaker_1 ,dataset_speaker_2 , n_frames = 128):

    num_samples = min(len(dataset_speaker_1 ), len(dataset_speaker_2))
    train_data_A_idx = np.arange(len(dataset_speaker_1 ))
    train_data_B_idx = np.arange(len(dataset_speaker_2))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()
    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B





def train(speaker_1, speaker_2, model_dir, model_name, random_seed,speaker_1_validation, speaker_2_validation, output_dir,
tensorboard_log_dir,num_epoch,batch_size,g_learning_rate ,d_learning_rate,sr,mcep,fp,fn):

    np.random.seed(random_seed)
    lambda_reconstruction= 8
    lambda_identity_tracing = 5
    print('feature extraction from speaker audio files started:::::: \n')
    start_time = time.time()
    
    #librosa.core.load(path, sr=16000, mono=True, offset=0.0, duration=None, dtype=<class ‘numpy.float32’>, res_type=’kaiser_best’)

    time_series_1 = librosa_load(speaker_1, sr)
    time_series_2 = librosa_load(speaker_2, sr)

    #encode data as per WORLD analysis

    f0s_speaker_1, timeaxes_speaker_1, sps_speaker_1, aps_speaker_1, coded_sps_speaker_1 = encode_data(time_series_1, sr , fp , mcep)
    f0s_speaker_2, timeaxes_speaker_2, sps_speaker_2, aps_speaker_2, coded_sps_speaker_2 = encode_data(time_series_2, sr , fp , mcep)
    
    #mean and standard deviation of logarithmic fundamental frequency    

    log_f0s_mean_speaker_1, log_f0s_std_speaker_1 = logf0_statistics(f0s_speaker_1)
    log_f0s_mean_speaker_2, log_f0s_std_speaker_2 = logf0_statistics(f0s_speaker_2)

    #transpose of coded_sps

    coded_sps_transposed_speaker_1 = transpose_coded_sps(coded_sps_speaker_1)
    coded_sps_transposed_speaker_2 = transpose_coded_sps(coded_sps_speaker_2)

    #normalized coded sps

    coded_sps_speaker_1_norm, coded_sps_speaker_1_mean, coded_sps_speaker_1_std = normalized_coded_sps(coded_sps_transposed_speaker_1)
    coded_sps_speaker_2_norm, coded_sps_speaker_2_mean, coded_sps_speaker_2_std = normalized_coded_sps(coded_sps_transposed_speaker_2)
    
    print(':::coded_sps_norm of speaker_1 & coded_sps_norm of speaker_2 are used for TF_GAN model training::: \n')
    print('coded_sps_norm of speaker_1 \n')
    print(coded_sps_speaker_1_norm)
    print('coded_sps_norm of speaker_2 \n')
    print(coded_sps_speaker_2_norm)
    

   #saving pre-processed data in npz format:::::::::::::
   
  
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_speaker_1 = log_f0s_mean_speaker_1, std_speaker_1 = log_f0s_std_speaker_1, mean_speaker_2 = log_f0s_mean_speaker_2, std_speaker_2 = log_f0s_std_speaker_2)
    
    print('saved the mean and standard deviation of logarithmic fundamental frequency values in npz format \n')
    
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_speaker_1 = coded_sps_speaker_1_mean, std_speaker_1 = coded_sps_speaker_1_std, mean_speaker_2 = coded_sps_speaker_2_mean, std_speaker_2 = coded_sps_speaker_2_std)
    
    print('saved the mean and standard deviation of mcep values in npz format \n')

    if speaker_1_validation is not None:
        validation_speaker_1_output_dir = os.path.join(output_dir, 'generated_fake_audio_samples_of_speaker_1')
        if not os.path.exists(validation_speaker_1_output_dir):
            os.makedirs(validation_speaker_1_output_dir)

    if speaker_2_validation is not None:
        validation_speaker_2_output_dir = os.path.join(output_dir, 'generated_fake_audio_samples_of_speaker_2')
        if not os.path.exists(validation_speaker_2_output_dir):
            os.makedirs(validation_speaker_2_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

   

    print('total time required for complete the feature extraction from speaker audio files  : %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))


    #building the TF-GAN model

    model = TF_GAN_model(num_features = mcep)


    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        

        start_time_epoch = time.time()

        dataset_speaker_1, dataset_speaker_2 = sample_train_data(dataset_speaker_1 = coded_sps_speaker_1_norm, dataset_speaker_2 = coded_sps_speaker_2_norm, n_frames = fn)

        n_samples = dataset_speaker_1.shape[0]

        for iteration in range(n_samples // batch_size):

            num_iterations = n_samples // batch_size * epoch + iteration

            if num_iterations > 20000:
               lambda_identity_tracing = 0
            if num_iterations > 200000:
                lambda_reconstruction= 10

            start =iteration * batch_size
            end = (iteration + 1) * batch_size

            generator_loss, discriminator_loss = model.train(input_A = dataset_speaker_1[start:end], input_B = dataset_speaker_2[start:end], lambda_reconstruction = lambda_reconstruction, lambda_identity_tracing = lambda_identity_tracing, generator_learning_rate = g_learning_rate, discriminator_learning_rate = d_learning_rate)

            if epoch==0:
                phi_G_x=generator_loss
                phi_D_y=discriminator_loss
            else:
                intermediate_x=phi_G_x
                intermediate_y=phi_D_y
                phi_G_x=generator_loss
                phi_D_y=discriminator_loss
                rate_of_change_of_G_x=0.05*(abs(intermediate_x-phi_G_x))
                rate_of_change_of_D_y=abs(intermediate_y-phi_D_y)
                if rate_of_change_of_G_x>rate_of_change_of_D_y:
                    g_learning_rate=0.000001+g_learning_rate
                    d_learning_rate=abs(d_learning_rate - 0.0000001)
                else:
                    d_learning_rate=0.000001+d_learning_rate
                    g_learning_rate=abs(g_learning_rate - 0.0000001)

            generator_loss, discriminator_loss = model.train(input_A = dataset_speaker_1[start:end], input_B = dataset_speaker_2[start:end], lambda_reconstruction = lambda_reconstruction, lambda_identity_tracing = lambda_identity_tracing, generator_learning_rate = g_learning_rate, discriminator_learning_rate = d_learning_rate)

            if  iteration % 50 == 0:
               
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(num_iterations, g_learning_rate, d_learning_rate, generator_loss, discriminator_loss))

        model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if speaker_1_validation is not None:
            if epoch % 50 == 0:
                print('Generating converted speaker data beta from  First Speaker...')
                for file in os.listdir(speaker_1_validation):
                    filepath = os.path.join(speaker_1_validation, file)
                    wav, _ = librosa.load(filepath, sr = sr, mono = True)
                    wav = wav_padding(wav = wav, sr = sr, frame_period = fp, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(data = wav, sr = sr, fp= fp)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_speaker_1, std_log_src = log_f0s_std_speaker_1, mean_log_target = log_f0s_mean_speaker_2, std_log_target = log_f0s_std_speaker_2)
                    coded_sp = world_encode_spectral_envelop(sp = sp, sr = sr, mcep = mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_speaker_1_mean) / coded_sps_speaker_1_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'alpha2beta')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_speaker_2_std + coded_sps_speaker_2_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, sr = sr)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, sr = sr, fp = fp)
                    librosa.output.write_wav(os.path.join(speaker_1_validation, os.path.basename(file)), wav_transformed, sr)

        if speaker_2_validation is not None:
            if epoch % 50 == 0:
                print('Generating converted speaker data beta from  Second Speaker...')
                for file in os.listdir(speaker_2_validation):
                    filepath = os.path.join(speaker_2_validation, file)
                    wav, _ = librosa.load(filepath, sr = sr, mono = True)
                    wav = wav_padding(wav = wav, sr = sr, frame_period = fp, multiple = 4)
                    f0, timeaxis, sp, ap = world_decompose(data = wav, sr = sr, fp= fp)
                    f0_converted = pitch_conversion(f0 = f0, mean_log_src = log_f0s_mean_speaker_2, std_log_src = log_f0s_std_speaker_2, mean_log_target = log_f0s_mean_speaker_1, std_log_target =log_f0s_std_speaker_1)
                    coded_sp = world_encode_spectral_envelop(sp = sp, sr = sr,mcep = mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_speaker_2_mean) /  coded_sps_speaker_2_std
                    coded_sp_converted_norm = model.test(inputs = np.array([coded_sp_norm]), direction = 'beta2alpha')[0]
                    coded_sp_converted = coded_sp_converted_norm * coded_sps_speaker_1_std + coded_sps_speaker_1_mean
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp = coded_sp_converted, sr = sr)
                    wav_transformed = world_speech_synthesis(f0 = f0_converted, decoded_sp = decoded_sp_converted, ap = ap, sr = sr, fp = fp)
                    librosa.output.write_wav(os.path.join(speaker_2_validation, os.path.basename(file)), wav_transformed, sr)

