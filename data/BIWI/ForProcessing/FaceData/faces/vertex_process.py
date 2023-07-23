import numpy as np
import os
import pandas as pd

input_path = "./"
output_path = "../../../vertices_npy/"
output_file = ""

subjects = [ name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name)) ] 

#print(subjects)

for subject in subjects:
    seqs = [ name for name in os.listdir(input_path+"/"+subject) if os.path.isdir(os.path.join(input_path+"/"+subject, name)) ]
#    print(seqs)
    for seq in seqs:
        print("now getting subject ", subject, " and seq ", seq)
        frame_files = os.listdir(input_path+"/"+subject+"/"+seq)
        if seq[0]=='e':
            new_seq = int(seq[1:3]) + 40
            output_file = subject+"_"+str(new_seq)+".npy"
        else:
            output_file = subject+"_"+seq+".npy"
        full_output_path = output_path+output_file
        empty_arr = np.empty((0,70110))
        for file in frame_files:
            if file.endswith(".csv"):
#                print(file)
                df=pd.read_csv(input_path+"/"+subject+"/"+seq+"/"+file, sep=',',header=None)
                df = df.iloc[1: , :]
                frame = df.to_numpy()
                frame = frame.flatten()
                frame = np.expand_dims(frame, axis=0)
                empty_arr = np.append(empty_arr,frame,axis=0)
        np.save(full_output_path, empty_arr)
        print("vertice shape: ", empty_arr.shape)
        print("saved to ", full_output_path)

templates_orig = np.load("../../../templates_orig.pkl", allow_pickle=True)
templates_scaled = np.load("../../../templates_scaled.pkl", allow_pickle=True)


path = "../../../vertices_npy/"

npys = [ name for name in os.listdir(path) if name.endswith('.npy') ]

for npy in npys:
    file_path = path + npy
    subject = npy.split('_')[0]
    subject_template = templates_orig[subject]
    
    subject_x_mean = subject_template[:,0].mean()
    subject_x_max = subject_template[:,0].max()
    subject_x_min = subject_template[:,0].min()
    
    subject_y_mean = subject_template[:,1].mean()
    subject_y_max = subject_template[:,1].max()
    subject_y_min = subject_template[:,1].min()
    
    subject_z_mean = subject_template[:,2].mean()
    subject_z_max = subject_template[:,2].max()
    subject_z_min = subject_template[:,2].min()
    
#     print(subject)
    seq = np.load(file_path, allow_pickle=True).astype(float)
#     print(seq.shape)
    seq = np.reshape(seq,(-1,70110//3,3))
    for f in range(seq.shape[0]):
        frame = seq[f,:,:]
        X = (frame[:,0]-subject_x_mean)/(subject_x_max-subject_x_min)
        Y = (frame[:,1]-subject_y_mean)/(subject_y_max-subject_y_min)
        Z = (frame[:,2]-subject_z_mean)/(subject_z_max-subject_z_min)
        frame[:,0] = X
        frame[:,1] = Y
        frame[:,2] = Z
        
        seq[f,:,:] = frame
    seq = seq.reshape(seq.shape[0],seq.shape[1]*seq.shape[2])
#     print(seq.shape)
    np.save(file_path, seq)