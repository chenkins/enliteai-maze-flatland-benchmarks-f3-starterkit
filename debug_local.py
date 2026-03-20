import pytest
from flatland.trajectories.policy_runner import generate_trajectory_from_policy


def main():
    with pytest.raises(SystemExit) as e_info:
        generate_trajectory_from_policy([
            "--data-dir", "/tmp",
            "--policy", "my_orga.policy.maze_policy.maze_xgboost_policy.MazeXGBoostPolicy",
            "--obs-builder", "flatland.envs.observations.FullEnvObservation",
            "--ep-id", "dummy",
            "--env-path", "./environments_v2/Test_00/Level_0.pkl"])
    assert e_info.value.code == 0


if __name__ == '__main__':
  main()
