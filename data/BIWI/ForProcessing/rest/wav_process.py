import os
import shutil

wav_in = "./audio/"
wav_out = "../../wav/"

wavs = os.listdir(wav_in)

for wav in wavs:
    input_path = wav_in + wav
#     print(input_path)
    if wav.endswith('_cut.wav'):
#        print(wav)
        filename = wav.split('_')
        sub = filename[0]
        seq = filename[1]
#        print(sub)
#        print(seq)
        if seq[0]=='e':
#            print(seq)
            new_seq = int(seq[1:3]) + 40
#            print(new_seq)
            output_path = wav_out + sub + "_" + str(new_seq) + ".wav"
            shutil.copy2(input_path, output_path)
#            os.rename(input_path, output_path)
        else:
            output_path = wav_out + sub + "_" + seq + ".wav"
            shutil.copy2(input_path, output_path)