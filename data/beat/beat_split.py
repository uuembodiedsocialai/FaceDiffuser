import json
import os
import scipy.io.wavfile as wav
import numpy as np
import pickle

seq_save_path = "vertices_npy"
audio_save_path = "wav"

emotion_ranges = {
    (0, 64): 0,
    (65, 72): 1,
    (73, 80): 2,
    (81, 86): 3,
    (87, 94): 4,
    (95, 102): 5,
    (103, 110): 6,
    (111, 118): 7
}

emotion_mapping = {
    0: "neutral",
    1: "happiness",
    2: "anger",
    3: "sadness",
    4: "contempt",
    5: "surprise",
    6: "fear",
    7: "disgust"
}

language_mapping = {
    0: "english",
    1: "english_conversation",
    2: "chinese",
    3: "chinese_conversation",
    4: "spanish",
    5: "spanish_conversation",
    6: "japanese",
    7: "japanese_convesation"
}


def helper_seq(path, actor_name, idx, emotion_category=0):

    with open(path) as f:
        anim_data = json.load(f)

    seq_len = 60 * 10 # 10 seconds at 60 FPS
    nr_seq = len(anim_data['frames']) // seq_len

    fs, audio = wav.read(f"{path[:-5]}.wav")

    for seq_idx in range(nr_seq):
        curr_seq = []

        # downsample from 60 fps to 30 fps
        for frame in anim_data['frames'][seq_idx * seq_len:(seq_idx + 1) * seq_len:2]:
            curr_frame = frame['weights']
            curr_seq.append(curr_frame)

        seq = np.array(curr_seq)
        np.save(os.path.join(seq_save_path, f"{actor_name}_{emotion_category}_{idx+1:02d}.npy"), seq)
        wav.write(os.path.join(audio_save_path, f"{actor_name}_{emotion_category}_{idx+1:02d}.wav"), fs, audio[fs * 10 * seq_idx:fs * 10 *(seq_idx + 1)])
        idx += 1

    return idx


def create_subject_sequences(s):
    seq_idx = 0
    actor_name = os.path.basename(s)
    print(actor_name)
    for x in os.scandir(s):
        if x.path.endswith(".json"):
            category = os.path.basename(x.path).split('.')[0].split('_')
            emotion_category = int(category[3])
            emotion_category = next((v for (a, b), v in emotion_ranges.items() if a <= emotion_category < b), 0)
            language = language_mapping[int(category[2])]
            if language == 'english': # only use english speech sequences
                seq_idx = helper_seq(x.path, actor_name, seq_idx, emotion_category)

    return actor_name


if __name__ == '__main__':

    path_to_dataset = ''
    subjects = [x.path for x in os.scandir(path_to_dataset)]
    actors = []
    for s in subjects:
        actors.append(create_subject_sequences(s))

    # template is not used since the neutral face is a 0 vector
    # generate it just for code generalization purposes
    template = {}
    for actor in actors:
        template[actor] = np.zeros((1, 51))

    with open("templates.pkl", 'wb') as handle:
        pickle.dump(template, handle, protocol=pickle.HIGHEST_PROTOCOL)