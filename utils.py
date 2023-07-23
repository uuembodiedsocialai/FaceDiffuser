import matplotlib.pyplot as plt
import pickle

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

with open('data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
    RIG_SCALER = pickle.load(f)


def plot_losses(train_losses, val_losses, save_name="losses"):
    print(train_losses)
    print(val_losses)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"{save_name}.png")
    plt.close()


def create_gaussian_diffusion(args):
    # default params
    sigma_small = True
    predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diff_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )