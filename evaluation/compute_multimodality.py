import numpy as np
import argparse
import os
import pickle


def get_mean_motion(seq):
    motion_vectors = np.linalg.norm(seq[1:, :, :] - seq[:-1, :, :], axis=2)
    # compute the mean and standard deviation of the motion vectors
    mean_motion = np.mean(motion_vectors, axis=0)
    return mean_motion


def main():
    # Computes the multimodality of the dataset. Inspired by the metric used in MDM.
    # In the end this metric was not used 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--path", type=str, default="/scratch/2176327/data/BIWI/vertices_npy/")
    parser.add_argument("--diversity_times", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="BIWI")
    args = parser.parse_args()

    train_subject_list = args.train_subjects.split(" ")

    if args.dataset == "BIWI":
        sentence_list = ["e" + str(i).zfill(2) for i in range(37, 41)]
    elif args.dataset == 'multiface':
        nr_vertices = 6172
        sentence_list = [str(i) for i in range(46, 51)]


        with open(args.templates_path, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        with open(os.path.join(args.region_path, "weighted_mouth_mask.txt")) as f:
            maps = [float(line.strip()) for line in f if line]
            mouth_map = []
            for idx, value in enumerate(maps):
                if value > 0.1:
                    mouth_map.append(idx)

        with open(os.path.join(args.region_path, "forehead_mask.txt")) as f:
            maps = [float(line.strip()) for line in f if line]
            upper_map = []
            for idx, value in enumerate(maps):
                if value > 0.1:
                    upper_map.append(idx)
    elif args.dataset == 'vocaset':
        # TODO
        pass

    cnt = 0
    vertices_gt_all = []
    mean_motion_all = []
    total_dist = 0
    total_seqs = 0
    for subject in train_subject_list:
        for sentence in sentence_list:

            all_samples = []
            for sample_idx in range(1, args.diversity_times + 1):
                anim_path = os.path.join(args.path, subject + "_" + sentence + "_condition_" + subject + "_" + str(sample_idx) + ".npy")

                if not os.path.exists(anim_path):
                    print(anim_path)
                    continue

                vertices = np.load(anim_path)
                all_samples.append(vertices)

            all_samples = np.stack(all_samples)

            # generate comparison pairs between any two animations
            comp_indices = [(i, j) for i in range(args.diversity_times) for j in range(args.diversity_times) if i != j]
            first_indices, second_indices = zip(*comp_indices)
            first_indices = list(first_indices)
            second_indices = list(second_indices)

            dist = np.linalg.norm(
                all_samples[first_indices] - all_samples[second_indices],
            )
            total_dist += dist
            total_seqs += 1
            print(dist)

    print('Multimodality: {:.4e}'.format(total_dist / total_seqs))


if __name__ == "__main__":
    main()