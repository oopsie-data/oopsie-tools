# ruff: noqa

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from pathlib import Path

from aloha.constants import FOLLOWER_GRIPPER_JOINT_OPEN
from einops import rearrange
import numpy as np
import torch
from torchvision import transforms
import tyro

from detr.models.latent_model import Latent_Model_Transformer
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from utils import set_seed

# =======================================
# ===== OopsieData project specific =====
# =======================================
from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_tools.utils.robot_profile.robot_profile import (
    act_plus_plus_robot_profile_path,
    load_robot_profile,
)
# =======================================


@dataclasses.dataclass
class Args:
    # ACT++ checkpoint / policy parameters
    ckpt_dir: str = ""
    policy_class: str = "ACT"  # choices: ACT, Diffusion, CNNMLP
    episode_len: int = 400
    ckpt_name: str = "policy_best.ckpt"
    temporal_agg: bool = False
    use_vq: bool = False
    vq_class: int | None = None
    vq_dim: int | None = None
    chunk_size: int | None = None
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    kl_weight: int = 10
    actuator_network_dir: str | None = None
    history_len: int | None = None
    future_len: int | None = None
    prediction_len: int | None = None

    # =======================================
    # ===== OopsieData project specific =====
    # =======================================
    data_root_dir: Path = Path("./data")
    resume_session_name: str | None = None
    operator_name: str = "operator"
    robot_profile: Path = dataclasses.field(
        default_factory=act_plus_plus_robot_profile_path
    )
    # =======================================


def main(args: Args):
    set_seed(1)

    # =======================================
    # ===== OopsieData project specific =====
    # =======================================
    robot_profile = load_robot_profile(args.robot_profile)
    camera_names = robot_profile.camera_names
    state_dim = len(robot_profile.robot_state_joint_names)
    control_freq = robot_profile.control_freq
    # =======================================

    lr_backbone = 1e-5
    backbone = "resnet18"

    if args.policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": 1e-5,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "vq": args.use_vq,
            "vq_class": args.vq_class,
            "vq_dim": args.vq_dim,
            "action_dim": 16,
        }
    elif args.policy_class == "Diffusion":
        policy_config = {
            "lr": 1e-5,
            "camera_names": camera_names,
            "action_dim": 16,
            "observation_horizon": 1,
            "action_horizon": 8,
            "prediction_horizon": args.chunk_size,
            "num_queries": args.chunk_size,
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
        }
    elif args.policy_class == "CNNMLP":
        policy_config = {
            "lr": 1e-5,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "vq": False,
            "action_dim": 16,
        }
    else:
        raise NotImplementedError

    config = {
        "ckpt_dir": args.ckpt_dir,
        "episode_len": args.episode_len,
        "state_dim": state_dim,
        "control_freq": control_freq,
        "policy_class": args.policy_class,
        "policy_config": policy_config,
        "temporal_agg": args.temporal_agg,
        "camera_names": camera_names,
        "real_robot": True,
    }

    # =======================================
    # ===== OopsieData project specific =====
    # =======================================
    episode_recorder = EpisodeRecorder(
        robot_profile=robot_profile,
        data_root_dir=args.data_root_dir,
        resume_session_name=args.resume_session_name,
        operator_name=args.operator_name,
    )
    # =======================================

    eval_bc(config, args.ckpt_name, episode_recorder)


def eval_bc(config, ckpt_name, episode_recorder: EpisodeRecorder):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    temporal_agg = config["temporal_agg"]
    control_freq = config["control_freq"]
    vq = policy_config.get("vq", False)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = policy_config["vq_dim"]
        vq_class = policy_config["vq_class"]
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, "latent_model_last.ckpt")
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f"Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}")
    else:
        print(f"Loaded: {ckpt_path}")

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    if policy_class == "Diffusion":
        post_process = lambda a: (
            ((a + 1) / 2) * (stats["action_max"] - stats["action_min"]) + stats["action_min"]
        )
    else:
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    from aloha.real_env import make_real_env
    from aloha.robot_utils import move_grippers
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

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    while True:
        # =======================================
        # ===== OopsieData project specific =====
        # =======================================
        episode_recorder.reset_episode_recorder()
        instruction = input("Enter instruction: ")
        # =======================================

        ts = env.reset()

        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, 16]
            ).cuda()

        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / control_freq
            culmulated_delay = 0
            for t in range(max_timesteps):
                time1 = time.time()

                obs = ts.observation
                if "images" in obs:
                    obs_images = obs["images"]
                else:
                    obs_images = {"main": obs["image"]}
                qpos_numpy = np.array(obs["qpos"])
                qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                if t % query_frequency == 0:
                    curr_image = get_image(
                        ts, camera_names, rand_crop_resize=(policy_class == "Diffusion")
                    )

                if t == 0:
                    # warm up
                    for _ in range(10):
                        policy(qpos, curr_image)
                    print("network warm up done")
                    time1 = time.time()

                if policy_class == "ACT":
                    if t % query_frequency == 0:
                        if vq:
                            if t == 0:
                                for _ in range(10):
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                            vq_sample = latent_model.generate(1, temperature=1, x=None)
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
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
                        all_time_actions[[t], t : t + num_queries - BASE_DELAY] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif policy_class == "Diffusion":
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
                elif policy_class == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                else:
                    raise NotImplementedError

                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]

                # =======================================
                # ===== OopsieData project specific =====
                # =======================================
                record_observation = {
                    "joint_position": np.asarray(qpos_numpy[:7], dtype=np.float32),
                    "gripper_position": np.asarray(
                        [qpos_numpy[7] if qpos_numpy.shape[0] > 7 else 0.0],
                        dtype=np.float32,
                    ),
                }
                for cam_name in camera_names:
                    if cam_name in obs_images:
                        record_observation[cam_name] = obs_images[cam_name]
                episode_recorder.record_step(
                    observation=record_observation,
                    action={"joint_position": np.asarray(target_qpos, dtype=np.float32)},
                )
                # =======================================

                if real_robot:
                    ts = env.step(target_qpos, base_action)
                else:
                    ts = env.step(target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += duration - DT
                    print(
                        f"Warning: step duration: {duration:.3f} s at step {t} longer than DT: "
                        f"{DT} s, culmulated delay: {culmulated_delay:.3f} s"
                    )

            print(f"Avg fps {max_timesteps / (time.time() - time0)}")

        if real_robot:
            move_grippers(
                [env.follower_bot_left, env.follower_bot_right],
                [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
                moving_time=0.5,
            )
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f"qpos_{log_id}.npy"), qpos_history_raw)

        # record success annotation
        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 1.0, n for 0.0), or 1.0/0.0 for success/failure: "
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success)
            if not success == 0.0 and not success == 1.0:
                print(f"Success must be 1.0 or 0.0 but got: {success}")

        # =======================================
        # ===== OopsieData project specific =====
        # =======================================
        episode_recorder.finish_rollout(instruction=instruction, success=success)
        # =======================================

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


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
            int(original_size[0] * (1 - ratio) / 2) : int(original_size[0] * (1 + ratio) / 2),
            int(original_size[1] * (1 - ratio) / 2) : int(original_size[1] * (1 + ratio) / 2),
        ]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
