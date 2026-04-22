import argparse
from copy import deepcopy
import os
import pickle
from pathlib import Path
import time

from aloha.constants import FPS, FOLLOWER_GRIPPER_JOINT_OPEN, TASK_CONFIGS
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from detr.models.latent_model import Latent_Model_Transformer
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils import (
    load_data,
    compute_dict_mean,
    set_seed,
)

from oopsie_tools.annotation_tool.rollout_annotator import WebRolloutAnnotator


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def _extract_video_views(image_list, camera_names):
    """Extract per-camera frame lists from image_list as a dict."""
    return {cam: [d[cam] for d in image_list if cam in d] for cam in camera_names}


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]

    # get task parameters
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    # num_episodes = task_config['num_episodes']
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "vq": args["use_vq"],
            "vq_class": args["vq_class"],
            "vq_dim": args["vq_dim"],
            "action_dim": 16,
        }
    elif policy_class == "Diffusion":
        policy_config = {
            "lr": args["lr"],
            "camera_names": camera_names,
            "action_dim": 16,
            "observation_horizon": 1,
            "action_horizon": 8,
            "prediction_horizon": args["chunk_size"],
            "num_queries": args["chunk_size"],
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "vq": False,
            "action_dim": 16,
        }
    else:
        raise NotImplementedError

    actuator_config = {
        "actuator_network_dir": args["actuator_network_dir"],
        "history_len": args["history_len"],
        "future_len": args["future_len"],
        "prediction_len": args["prediction_len"],
    }

    config = {
        "num_steps": num_steps,
        "validate_every": validate_every,
        "save_every": save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": resume_ckpt_path,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not args["onscreen_render"],
        "actuator_config": actuator_config,
        "policy_id": args["policy_id"],
    }

    if is_eval:
        ckpt_names = [args["ckpt_name"]]
        rollout_annotator = WebRolloutAnnotator(
            samples_dir=args["samples_dir"],
            policy_name=args["policy_name"],
            policy_id=args["policy_id"],
            camera_names=camera_names,
            annotator_port=args["annotator_port"],
            wait_for_annotation=not args["no_wait_for_annotation"],
        )
        for ckpt_name in ckpt_names:
            eval_bc(config, ckpt_name, rollout_annotator)
        exit()

    # training (unchanged)
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes=None,
        camera_names=camera_names,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        sample_weights=sample_weights,
    )
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def eval_bc(config, ckpt_name, rollout_annotator: WebRolloutAnnotator):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "angle"
    vq = config["policy_config"]["vq"]
    actuator_config = config["actuator_config"]

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config["policy_config"]["vq_dim"]
        vq_class = config["policy_config"]["vq_class"]
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, "latent_model_last.ckpt")
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(
            f"Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}"
        )
    else:
        print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    if policy_class == "Diffusion":
        post_process = lambda a: (
            ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    from aloha.real_env import make_real_env  # requires aloha
    from aloha.robot_utils import move_grippers  # requires aloha
    from interbotix_common_modules.common_robot.robot import (
        create_interbotix_global_node,
        get_interbotix_global_node,
        robot_startup,
    )
    from interbotix_common_modules.common_robot.exceptions import InterbotixException

    try:
        node = get_interbotix_global_node()
    except:
        node = create_interbotix_global_node("aloha")
    env = make_real_env(node=node, setup_robots=True, setup_base=True)
    try:
        robot_startup(node)
    except InterbotixException:
        pass
    env_max_reward = 0

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    rollout_annotator.start()
    print("Press Ctrl+C to end the session.")
    try:
        while True:
            instruction = rollout_annotator.wait_for_task()
            rollout_annotator.start_rollout()

            ts = env.reset()

            ### onscreen render
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(
                    env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                )
                plt.ion()

            # evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros(
                    [max_timesteps, max_timesteps + num_queries, 16]
                ).cuda()

            qpos_history_raw = np.zeros((max_timesteps, state_dim))
            image_list = []  # for visualization
            qpos_list = []
            target_qpos_list = []
            rewards = []
            with torch.inference_mode():
                time0 = time.time()
                DT = 1 / FPS
                culmulated_delay = 0
                for t in range(max_timesteps):
                    time1 = time.time()
                    ### update onscreen render and wait for DT
                    if onscreen_render:
                        image = env._physics.render(
                            height=480, width=640, camera_id=onscreen_cam
                        )
                        plt_img.set_data(image)
                        plt.pause(DT)

                    ### process previous timestep to get qpos and image_list
                    time2 = time.time()
                    obs = ts.observation
                    if "images" in obs:
                        image_list.append(obs["images"])
                    else:
                        image_list.append({"main": obs["image"]})
                    qpos_numpy = np.array(obs["qpos"])
                    qpos_history_raw[t] = qpos_numpy
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    if t % query_frequency == 0:
                        curr_image = get_image(
                            ts,
                            camera_names,
                            rand_crop_resize=(config["policy_class"] == "Diffusion"),
                        )

                    if t == 0:
                        # warm up
                        for _ in range(10):
                            policy(qpos, curr_image)
                        print("network warm up done")
                        time1 = time.time()

                    ### query policy
                    time3 = time.time()
                    if config["policy_class"] == "ACT":
                        if t % query_frequency == 0:
                            if vq:
                                if t == 0:
                                    for _ in range(10):
                                        vq_sample = latent_model.generate(
                                            1, temperature=1, x=None
                                        )
                                        print(
                                            torch.nonzero(vq_sample[0])[:, 1]
                                            .cpu()
                                            .numpy()
                                        )
                                vq_sample = latent_model.generate(
                                    1, temperature=1, x=None
                                )
                                all_actions = policy(
                                    qpos, curr_image, vq_sample=vq_sample
                                )
                            else:
                                all_actions = policy(qpos, curr_image)
                            if real_robot:
                                all_actions = torch.cat(
                                    [
                                        all_actions[:, :-BASE_DELAY, :-2],
                                        all_actions[:, BASE_DELAY:, -2:],
                                    ],
                                    dim=2,
                                )
                        if temporal_agg:
                            all_time_actions[[t], t : t + num_queries - BASE_DELAY] = (
                                all_actions
                            )
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(
                                actions_for_curr_step != 0, axis=1
                            )
                            actions_for_curr_step = actions_for_curr_step[
                                actions_populated
                            ]
                            k = 0.01
                            exp_weights = np.exp(
                                -k * np.arange(len(actions_for_curr_step))
                            )
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = (
                                torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            )
                            raw_action = (actions_for_curr_step * exp_weights).sum(
                                dim=0, keepdim=True
                            )
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config["policy_class"] == "Diffusion":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                            if real_robot:
                                all_actions = torch.cat(
                                    [
                                        all_actions[:, :-BASE_DELAY, :-2],
                                        all_actions[:, BASE_DELAY:, -2:],
                                    ],
                                    dim=2,
                                )
                        raw_action = all_actions[:, t % query_frequency]
                    elif config["policy_class"] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                        all_actions = raw_action.unsqueeze(0)
                    else:
                        raise NotImplementedError

                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action[:-2]

                    base_action = action[-2:]

                    # step the environment
                    record_observation = {
                        "joint_position": np.asarray(qpos_numpy[:7], dtype=np.float32),
                        "gripper_position": np.asarray(
                            [qpos_numpy[7] if qpos_numpy.shape[0] > 7 else 0.0],
                            dtype=np.float32,
                        ),
                    }
                    obs_images = obs.get("images", {}) if isinstance(obs, dict) else {}
                    for cam_name in camera_names:
                        if cam_name in obs_images:
                            record_observation[cam_name] = obs_images[cam_name]
                    rollout_annotator.record_step(
                        observation=record_observation,
                        action={
                            "joint_position": np.asarray(target_qpos, dtype=np.float32)
                        },
                    )

                    if real_robot:
                        ts = env.step(target_qpos, base_action)
                    else:
                        ts = env.step(target_qpos)

                    ### for visualization
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(ts.reward)
                    duration = time.time() - time1
                    sleep_time = max(0, DT - duration)
                    time.sleep(sleep_time)
                    if duration >= DT:
                        culmulated_delay += duration - DT
                        print(
                            (
                                f"Warning: step duration: {duration:.3f} s at step {t} longer than DT: "
                                f"{DT} s, culmulated delay: {culmulated_delay:.3f} s"
                            )
                        )

                print(f"Avg fps {max_timesteps / (time.time() - time0)}")
                plt.close()
            if real_robot:
                move_grippers(
                    [env.follower_bot_left, env.follower_bot_right],
                    [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
                    moving_time=0.5,
                )  # open
                # save qpos_history_raw
                log_id = get_auto_index(ckpt_dir)
                np.save(os.path.join(ckpt_dir, f"qpos_{log_id}.npy"), qpos_history_raw)
                plt.figure(figsize=(10, 20))
                for i in range(state_dim):
                    plt.subplot(state_dim, 1, i + 1)
                    plt.plot(qpos_history_raw[:, i])
                    if i != state_dim - 1:
                        plt.xticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(ckpt_dir, f"qpos_{log_id}.png"))
                plt.close()

            # Extract per-camera frames and annotate
            videos = _extract_video_views(image_list, camera_names)
            rollout_annotator.finish_rollout(
                instruction=instruction,
                videos=videos,
                t_step=max_timesteps - 1,
            )

    except KeyboardInterrupt:
        pass


# ---- helpers (unchanged from run_policy.py) ----


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    if rand_crop_resize:
        print("rand crop resize is used!")
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[
            ...,
            int(original_size[0] * (1 - ratio) / 2) : int(
                original_size[0] * (1 + ratio) / 2
            ),
            int(original_size[1] * (1 - ratio) / 2) : int(
                original_size[1] * (1 + ratio) / 2
            ),
        ]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config["resume_ckpt_path"] is not None:
        loading_status = policy.deserialize(torch.load(config["resume_ckpt_path"]))
        print(
            f"Resume policy from: {config['resume_ckpt_path']}, Status: {loading_status}"
        )
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for step in tqdm(range(num_steps)):
        # validation
        if step % validate_every == 0:
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break
                validation_summary = compute_dict_mean(validation_dicts)
                validation_history.append(validation_summary)
                epoch_val_loss = validation_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in validation_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(iter(train_dataloader))
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()
        train_history.append(forward_dict)
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, step, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)
    return best_ckpt_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_steps", action="store", type=int, help="num_steps", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)
    parser.add_argument("--validate_every", action="store", type=int, default=500)
    parser.add_argument("--save_every", action="store", type=int, default=500)
    parser.add_argument("--resume_ckpt_path", action="store", type=str, default=None)
    parser.add_argument(
        "--ckpt_name", action="store", type=str, default="policy_best.ckpt"
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--use_vq", action="store_true")
    parser.add_argument("--vq_class", action="store", type=int, default=None)
    parser.add_argument("--vq_dim", action="store", type=int, default=None)
    parser.add_argument("--chunk_size", action="store", type=int, default=None)
    parser.add_argument("--hidden_dim", action="store", type=int, default=512)
    parser.add_argument("--dim_feedforward", action="store", type=int, default=3200)
    parser.add_argument("--kl_weight", action="store", type=int, default=10)
    parser.add_argument(
        "--actuator_network_dir", action="store", type=str, default=None
    )
    parser.add_argument("--history_len", action="store", type=int)
    parser.add_argument("--future_len", action="store", type=int)
    parser.add_argument("--prediction_len", action="store", type=int)
    # Annotation parameters
    parser.add_argument("--samples_dir", type=Path, default=Path("./samples"))
    parser.add_argument("--policy_name", type=str, default="act")
    parser.add_argument(
        "--policy_id",
        type=str,
        default="act",
        help="Policy ID for session directory naming",
    )
    parser.add_argument(
        "--camera_names", type=str, nargs="+", default=["left", "right", "wrist"]
    )
    parser.add_argument("--annotator_port", type=int, default=5001)
    parser.add_argument("--no_wait_for_annotation", action="store_true")
    args = vars(parser.parse_args())
    main(args)
