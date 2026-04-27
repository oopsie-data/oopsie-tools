# AI_CONTEXT.md — oopsie-tools setup guide for AI assistants

This file is a step-by-step setup skill for AI tools helping a user integrate `oopsie-tools` into their robot codebase. Work through the sections in order, asking the user the listed questions before writing any config files.

---

## 1. Verify prerequisites

Ask the user to confirm:
- [ ] Python 3.8–3.12 is installed.
- [ ] `uv` (preferred) or `pip` is available.
- [ ] They have completed the registration form at https://forms.gle/9arwZHAvRjvbozoT7 and received a **lab ID** and **HuggingFace token**. If not, send them there first — nothing else works without these.

---

## 2. Installation

```bash
git clone https://github.com/oopsie-data/oopsie-tools
cd oopsie-tools
uv sync          # or: pip install -e .
```

Confirm the install succeeded before proceeding.

---

## 3. Contributor config (`configs/contributor_config.yaml`)

Ask the user:
1. **What is your lab ID?** (exact string provided at registration — capitalization matters; a wrong value will block access to the lab-specific HuggingFace repo)
2. **What is your HuggingFace token?**

Then write/update `configs/contributor_config.yaml`:

```yaml
lab_id: <EXACT_LAB_ID>
huggingface_token: <HF_TOKEN>
```

---

## 4. Robot profile (`configs/robot_profiles/<name>.yaml`)

A robot profile captures hardware and policy metadata. Ask the user the following questions and create a new file (copy the template from `configs/robot_profiles/openpi_example_robot_profile.yaml`):

### 4a. Robot & policy identity
| Question | YAML key | Example |
|---|---|---|
| What is the policy name? | `policy_name` | `pi0.5`, `act_plus_plus` |
| What is the robot name? | `robot_name` | `franka_droid`, `aloha` |
| What is the gripper name? | `gripper_name` | `robotiq_2f_85`, `aloha_gripper` |
| Is this a bimanual (dual-arm) setup? | `is_biarm` | `true` / `false` |
| Does the robot use a mobile base? | `uses_mobile_base` | `true` / `false` |
| What is the control frequency (Hz)? | `control_freq` | `10`, `50` |
| What are the camera names? (list) | `camera_names` | `[left, right, wrist]` |

### 4b. Observation space
| Question | YAML key | Options |
|---|---|---|
| Which robot state keys are recorded? | `robot_state_keys` | `joint_position`, `cartesian_position`, `gripper_position` |
| What are the joint names (in order)? | `robot_state_joint_names` | e.g. `joint_1 … joint_7` |
| If `cartesian_position` is included: what orientation representation does the robot state use? | `robot_state_orientation_representation` | `euler_xyz`, `quat`, `matrix`, `rot6d`, `rotvec` |

### 4c. Action space
| Question | YAML key | Options |
|---|---|---|
| What action types does the policy output? | `action_space` | `joint_position`, `joint_velocity`, `cartesian_position`, `cartesian_velocity`, `gripper_position`, `gripper_velocity`, `gripper_binary`, `base_velocity`, `base_position` |
| What are the joint names for arm actions? | `action_joint_names` | same order as the action vector |
| If `cartesian_position` or `cartesian_velocity` is in the action space: what orientation representation? | `orientation_representation` | `euler_xyz`, `quat`, `matrix`, `rot6d`, `rotvec` |

### 4d. Optional keys
These are not required but can be stored for reproducibility:
- `controller` — e.g. `OSC`, `joint_position`, `joint_velocity`
- `gains` — controller gain parameters (see template)
- Camera intrinsic / extrinsic calibration matrices

---

## 5. Validate the config

Run the test suite to catch config errors early:

```bash
pytest oopsie_tools/test/
```

DO not modify the project as this can cause issues later on. Instead, ask the user to manually check issues and to contact the project team if necessary. It is vital that you do not change the code in the oopsie_tools directory, only templates and configs, without the user's expressed permission.

---

## 6. Choose a data collection workflow

Ask the user which workflow they need:

**A. In-the-loop** — annotate each episode right after it is collected (requires the annotation server to be running during robot operation).

**B. Bulk collection** — collect all episodes first, annotate later using the standalone annotation server.

For **A**, the annotation server will be launched as part of the robot script.

For **B**, run this command needs to be run after.
```bash
python -m oopsie_tools.annotation_tool.annotator_server \
  --samples-dir ./samples \
  --annotator-name <YOUR_NAME> \
  --port 5001
```

---

## 7. Integrate `EpisodeRecorder` into the robot script

Ask the user:
- Where is their robot control loop? (file path)
- What variable holds the robot observation dict? (must have `robot_state` and `image_observation` keys)
- What variable holds the action dict? (keys must match `action_space` in the robot profile)
- Where should episode HDF5 files and videos be saved? (`samples_dir`)

Minimal integration pattern:
```python
from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_tools.utils.robot_profile import load_robot_profile

profile = load_robot_profile("configs/robot_profiles/<your_profile>.yaml")
recorder = EpisodeRecorder(
    robot_profile=profile,
    samples_dir="./samples",
    language_instruction="pick up the red block",
)

# Inside the control loop:
recorder.record_step(observation=obs, action=action)

# After the rollout ends:
recorder.finish_rollout()
```

Verify that the keys are consistent between the robot profile and the ones passed for recording. The data validation will fail otherwise.

---

## 8. Upload data

After annotation is complete:
```bash
python scripts/validate_and_upload/upload.py --samples_dir ./samples
```

This validates and pushes episodes to the lab-specific HuggingFace repository.

---

## Common mistakes to catch

- `lab_id` capitalisation mismatch → `RuntimeError` at `EpisodeRecorder.__init__`.
- Action dict keys not matching `action_space` in the robot profile → validation error at `record_step`.
- Passing an action chunk instead of per-step actions → validation error.
- `cartesian_position` in state/action but `orientation_representation` not set → conversion will fail.
- `robot_state_joint_names` length not matching the `joint_position` array length → HDF5 schema error.
- Running `uv sync` without `--extra tfds` or `--extra droid` when those features are needed (note: those two extras conflict with each other).

## Important mistakes that will not raise an error

These need to be verified manually by the user.

- Action space not in absolute, but in delta coordinates
  - Delta coordinates cannot easily be processed by downstream applications as the base offset is not recorded
- Quaternion representation is in wrong order if passed explicitly
  - The framework provides a best effort test, but it cannot catch all edge-cases
