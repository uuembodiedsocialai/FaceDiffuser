import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="/scratch/2176327/data/damm_rig_equal/vertices_npy_192/")
    args = parser.parse_args()


    cnt = 0
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    lve = 0
    num_seq = 0

    mouth_mask = list(range(94, 114)) + list(range(146, 178)) + list(range(183, 192))
    upper_mask = [x for x in range(192) if x not in mouth_mask]
    for file in os.listdir(args.pred_path):
        if file.endswith('.npy'):
            seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:2])

            gt_seq = np.load(os.path.join(args.gt_path, seq_name + ".npy"))

            pred_seq = np.load(os.path.join(args.pred_path, file))

            pred_seq = pred_seq[:gt_seq.shape[0], :]
            gt_seq = gt_seq[:pred_seq.shape[0], :]

            mve += np.linalg.norm(pred_seq - gt_seq, axis = 1).mean()
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            cnt += pred_seq.shape[0]

            L2_dis_upper = np.array([np.square(gt_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(pred_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)
            abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
            num_seq += 1

    print('Frame Number: {}'.format(cnt))

    print('Mean Vertex Error: {:.4e}'.format(mve / num_seq))
    print('Lip Vertex Error: {:.4e}'.format(lve / num_seq))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


def main_beat():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="/scratch/2176327/data/beat/vertices_npy")
    args = parser.parse_args()


    cnt = 0
    motion_std_difference = []
    abs_motion_std_difference = []

    mve = 0
    lve = 0
    num_seq = 0

    upper_mask = [
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 49, 50
    ]

    mouth_mask = [
        23, 25, 27, 28, 31, 37, 39, 40, 41, 42, 47, 48
    ]

    for file in os.listdir(args.pred_path):
        if file.endswith('.npy'):
            seq_name = "_".join(os.path.basename(file).split('.')[0].split('_')[:3])

            gt_seq = np.load(os.path.join(args.gt_path, seq_name + ".npy"))

            pred_seq = np.load(os.path.join(args.pred_path, file))

            pred_seq = pred_seq[:gt_seq.shape[0], :]
            gt_seq = gt_seq[:pred_seq.shape[0], :]

            mve += np.linalg.norm(pred_seq - gt_seq, axis = 1).mean()
            lve += np.linalg.norm(pred_seq[:, mouth_mask] - gt_seq[:, mouth_mask], axis=1).mean()

            cnt += pred_seq.shape[0]

            L2_dis_upper = np.array([np.square(gt_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(pred_seq[:, v]) for v in upper_mask])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0))
            L2_dis_upper = np.sum(L2_dis_upper, axis=1)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)
            abs_motion_std_difference.append(np.abs(gt_motion_std - pred_motion_std))
            num_seq += 1

    print('Frame Number: {}'.format(cnt))

    print('Mean Vertex Error: {:.4e}'.format(mve / num_seq))
    print('Lip Vertex Error: {:.4e}'.format(lve / num_seq))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))
    print('ABS FDD: {:.4e}'.format(sum(abs_motion_std_difference) / len(motion_std_difference)))


if __name__ == "__main__":
    main_beat()
