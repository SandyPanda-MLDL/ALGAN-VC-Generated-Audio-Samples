import os
import argparse


from train import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    #Speaker audio directory for training
    speaker_1_training_dir ='./training_data/speaker_1'
    speaker_2_training_dir ='./training_data/speaker_2'

    #Speaker audio directory for validation
    speaker_1_validation_dir ='./validation/speaker_1'
    speaker_2_validation_dir ='./validation/speaker_2'
    generated_data_directory = './output_samples'

    tfgan_model_directory ='./model/tfgan'
    tfgan_model_name ='tfgan.ckpt'
    tensorboard_log_directory = './tensorboard_log'
    batch_size=1
    random_seed= 1
    g_learning_rate= 0.0002
    d_learning_rate= 0.0001
    
   
    parser.add_argument('--speaker_1_dir', type = str, help = 'Audio directory of the first speaker', default = speaker_1_training_dir )
    parser.add_argument('--speaker_2_dir', type = str, help = 'Audio directory of the second speaker', default = speaker_2_training_dir )

    parser.add_argument('--speaker_1_validation_dir', type = str, help = 'vocal style conversion of speaker_1  during  training.', default =  speaker_1_validation_dir)
    parser.add_argument('--speaker_2_validation_dir', type = str, help = 'vocal style conversion of speaker_2  during  training.', default =  speaker_2_validation_dir)
    parser.add_argument('--generated_data_directory', type = str, help = 'output directory for saving the generated audio samples', default = generated_data_directory)
    
    parser.add_argument('--epoch_number', type = int, help = 'epoch_number', default = 5000)
    
    parser.add_argument('--tfgan_model_directory', type = str, help = 'tfgan_model_directory for saving the model.', default = tfgan_model_directory )
    parser.add_argument('--tfgan_model_name', type = str, help = 'tfgan_model_name', default = tfgan_model_name)

    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)

    parser.add_argument('--random_seed', type = int, help = 'random seed value', default = random_seed)
    parser.add_argument('--batch_size', type = int, help = 'mini_batch_size', default = batch_size)
    
    parser.add_argument('--g_learning_rate', type = int, help = 'generator_learning_rate', default = g_learning_rate)
    parser.add_argument('--d_learning_rate', type = int, help = 'discriminator_learning_rate', default = d_learning_rate)

    parser.add_argument('--sampling_rate', type = int, help = 'sampling_rate', default = 16000)
    parser.add_argument('--MCEPs', type = int, help = 'number of mel cepstral coefficients', default = 24)
    parser.add_argument('--fp', type = int, help = 'frame period', default = 5.0)
    parser.add_argument('--fn', type = int, help = 'frame number', default = 128)

    

    argv = parser.parse_args()

    speaker_1= argv.speaker_1_dir
    speaker_2= argv.speaker_2_dir
    
    speaker_1_validation = argv.speaker_1_validation_dir
    speaker_2_validation = argv.speaker_2_validation_dir
    
    num_epoch = argv.epoch_number

    output_dir = argv.generated_data_directory

    model_dir = argv.tfgan_model_directory
    model_name = argv.tfgan_model_name
    random_seed = argv.random_seed
    batch_size = argv.batch_size
     
    g_learning_rate=argv.g_learning_rate
    d_learning_rate=argv.d_learning_rate

    sr=argv.sampling_rate
    mcep=argv.MCEPs
    fp=argv.fp
    fn=argv.fn

    tensorboard_log_dir = argv.tensorboard_log_dir

    train(speaker_1, speaker_2, model_dir, model_name, random_seed,speaker_1_validation, speaker_2_validation, output_dir,
tensorboard_log_dir,num_epoch,batch_size,g_learning_rate,d_learning_rate,sr,mcep,fp,fn)
