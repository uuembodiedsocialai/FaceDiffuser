import argparse
import os
import pickle
import shutil

import pandas as pd
import torch
import numpy as np

from data_loader import get_dataloaders
from diffusion.resample import create_named_schedule_sampler
from tqdm import tqdm

from models import FaceDiff, FaceDiffBeat, FaceDiffDamm
from utils import *


def trainer_diff(args, train_loader, dev_loader, model, diffusion, optimizer, epoch=100, device="cuda"):
    train_losses = []
    val_losses = []

    save_path = os.path.join(args.save_path)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    iteration = 0

    for e in range(epoch + 1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            # for vocaset reduce the frame rate from 60 to 30
            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot = template.to(device=device), one_hot.to(device=device)

            loss = diffusion.training_losses(
                model,
                x_start=vertice,
                t=t,
                model_kwargs={
                    "cond_embed": audio,
                    "one_hot": one_hot,
                    "template": template,
                }
            )['loss']

            loss = torch.mean(loss)
            loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                del audio, vertice, template, one_hot
                torch.cuda.empty_cache()

            pbar.set_description(
                "(Epoch {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), iteration, np.mean(loss_log)))

        train_losses.append(np.mean(loss_log))

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # to gpu
            vertice = str(vertice[0])
            vertice = np.load(vertice, allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)

            # for vocaset reduce the frame rate from 60 to 30
            if args.dataset == 'vocaset':
                vertice = vertice[::2, :]
            vertice = torch.unsqueeze(vertice, 0)

            t, weights = schedule_sampler.sample(1, torch.device(device))

            audio, vertice = audio.to(device=device), vertice.to(device=device)
            template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

            train_subject = file_name[0].split("_")[0]
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]

                loss = diffusion.training_losses(
                    model,
                    x_start=vertice,
                    t=t,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                    }
                )['loss']

                loss = torch.mean(loss)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    one_hot = one_hot_all[:, iter, :]
                    loss = diffusion.training_losses(
                        model,
                        x_start=vertice,
                        t=t,
                        model_kwargs={
                            "cond_embed": audio,
                            "one_hot": one_hot,
                            "template": template,
                        }
                    )['loss']

                    loss = torch.mean(loss)
                    valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        val_losses.append(current_loss)
        if e == args.max_epoch or e % 25 == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'{args.model}_{args.dataset}_{e}.pth'))
            plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))
        print("epcoh: {}, current loss:{:.8f}".format(e + 1, current_loss))

    plot_losses(train_losses, val_losses, os.path.join(save_path, f"losses_{args.model}_{args.dataset}"))

    return model


@torch.no_grad()
def test_diff(args, model, test_loader, epoch, diffusion, device="cuda"):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, f'{args.model}_{args.dataset}_{epoch}.pth')))
    model = model.to(torch.device(device))
    model.eval()

    sr = 16000
    for audio, vertice, template, one_hot_all, file_name, emo_one_hot in test_loader:
        vertice = vertice_path = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        if args.dataset == 'vocaset':
            vertice = vertice[::2, :]
        vertice = torch.unsqueeze(vertice, 0)


        audio, vertice =  audio.to(device=device), vertice.to(device=device)
        template, one_hot_all = template.to(device=device), one_hot_all.to(device=device)

        num_frames = int(audio.shape[-1] / sr * args.output_fps)
        shape = (1, num_frames - 1, args.vertice_dim) if num_frames < vertice.shape[1] else vertice.shape

        train_subject = file_name[0].split("_")[0]
        vertice_path = os.path.split(vertice_path)[-1][:-4]
        print(vertice_path)

        if train_subject in train_subjects_list or args.dataset == 'beat':
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            one_hot = one_hot.to(device=device)

            for sample_idx in range(1, args.num_samples + 1):
                sample = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                        "guidance_weight": 0,
                    },
                    skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                sample = sample.squeeze()
                sample = sample.detach().cpu().numpy()

                if args.dataset == 'beat':
                    out_path = f"{vertice_path}.npy"
                else:
                    if args.num_samples != 1:
                        out_path = f"{vertice_path}_condition_{condition_subject}_{sample_idx}.npy"
                    else:
                        out_path = f"{vertice_path}_condition_{condition_subject}.npy"
                if 'damm' in args.dataset:
                    sample = RIG_SCALER.inverse_transform(sample)
                    np.save(os.path.join(args.result_path, out_path), sample)
                    df = pd.DataFrame(sample)
                    df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path, out_path), sample)

        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                one_hot = one_hot.to(device=device)

                # sample conditioned
                sample_cond = diffusion.p_sample_loop(
                    model,
                    shape,
                    clip_denoised=False,
                    model_kwargs={
                        "cond_embed": audio,
                        "one_hot": one_hot,
                        "template": template,
                        "guidance_weight": 0,
                    },
                    skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    device=device
                )
                prediction_cond = sample_cond.squeeze()
                prediction_cond = prediction_cond.detach().cpu().numpy()

                prediction = prediction_cond
                if 'damm' in args.dataset:
                    prediction = RIG_SCALER.inverse_transform(prediction)
                    df = pd.DataFrame(prediction)
                    df.to_csv(os.path.join(args.result_path, f"{vertice_path}.csv"), header=None, index=None)
                else:
                    np.save(os.path.join(args.result_path, f"{vertice_path}_condition_{condition_subject}.npy"), prediction)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=70110, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=512, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=512, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates.pkl",
                        help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=1, help='number of samples to generate per audio')
    args = parser.parse_args()

    assert torch.cuda.is_available()
    diffusion = create_gaussian_diffusion(args)

    if 'damm' in args.dataset:
        model = FaceDiffDamm(args)
    elif 'beat' in args.dataset:
        model = FaceDiffBeat(
                args,
                vertice_dim=args.vertice_dim,
                latent_dim=args.feature_dim,
                diffusion_steps=args.diff_steps,
                cond_dropout_type=args.cond_drop,
                gru_latent_dim=args.gru_dim,
                num_layers=args.gru_layers,
            )
    else:
        model = FaceDiff(
            args,
            vertice_dim=args.vertice_dim,
            latent_dim=args.feature_dim,
            diffusion_steps=args.diff_steps,
            cond_dropout_type=args.cond_drop,
            gru_latent_dim=args.gru_dim,
            num_layers=args.gru_layers,
        )
    print("model parameters: ", count_parameters(model))
    cuda = torch.device(args.device)

    model = model.to(cuda)
    dataset = get_dataloaders(args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model = trainer_diff(args, dataset["train"], dataset["valid"], model, diffusion, optimizer,
                         epoch=args.max_epoch, device=args.device)
    test_diff(args, model, dataset["test"], args.max_epoch, diffusion, device=args.device)


if __name__ == "__main__":
    main()