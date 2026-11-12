"""Frozen CleanRL Ant gait with local directional steering (no external forces)."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "pretrained/ppo_fix_continuous_action.cleanrl_model"


class AntDirectionController(nn.Module):
    def __init__(self, checkpoint=None, steering_gain=.6, max_steering=.4):
        super().__init__()
        checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing Ant gait checkpoint {checkpoint}. Run python -m distributed_swarm.scripts.download_ant_policy")
        self.actor = nn.Sequential(nn.Linear(27,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,8))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.actor.load_state_dict({k.removeprefix("actor_mean."):v for k,v in state.items() if k.startswith("actor_mean.")})
        self.register_buffer("obs_mean", state["obs_rms.mean"].float())
        self.register_buffer("obs_var", state["obs_rms.var"].float())
        self.steering_gain, self.max_steering = steering_gain, max_steering
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def local_observation(qpos, qvel, heading):
        """Express own Ant-v4 proprioception in a desired-heading world frame.

        Free-joint linear velocity is world-frame; its angular velocity is local
        to the body and must not be rotated as if it were world-frame.
        """
        q = np.asarray(qpos).copy()
        v = np.asarray(qvel).copy()
        c, s = np.cos(heading), np.sin(heading)
        v[0], v[1] = c*v[0]+s*v[1], -s*v[0]+c*v[1]
        w,x,y,z = q[3:7]
        ch, sh = np.cos(heading/2), np.sin(heading/2)
        q[3:7] = [ch*w+sh*z, ch*x+sh*y, ch*y-sh*x, ch*z-sh*w]
        return np.r_[q[2:],np.clip(v,-10.,10.)].astype(np.float32)

    def torques(self, qpos, qvel, direction):
        direction = np.asarray(direction,dtype=float)
        if direction.shape != (2,) or not np.isfinite(direction).all():
            raise ValueError("direction must contain two finite world-frame xy values")
        norm = np.linalg.norm(direction)
        if norm < .05:
            # Local posture hold for STOP, not a gait evaluated at an arbitrary heading.
            rest = np.array([0.,1.,0.,-1.,0.,-1.,0.,1.])
            hold = 2*(rest-np.asarray(qpos)[7:])-.15*np.asarray(qvel)[6:]
            # qpos joint order is hip1..hip4; actuator order starts at hip4.
            return np.clip(hold[[6,7,0,1,2,3,4,5]],-1,1).astype(np.float32)
        heading = np.arctan2(direction[1],direction[0])
        w,x,y,z = qpos[3:7]
        yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        error = np.arctan2(np.sin(heading-yaw),np.cos(heading-yaw))
        # The gait was trained to walk forward. Turn toward large heading
        # changes gradually so a 180-degree command does not destabilize it.
        reference_heading = yaw + np.clip(error,-.6,.6)
        obs = self.local_observation(qpos,qvel,reference_heading)
        with torch.inference_mode():
            normalized = ((torch.as_tensor(obs)-self.obs_mean)/torch.sqrt(self.obs_var+1e-8)).clamp(-10,10)
            action = self.actor(normalized).numpy()
        # Equal hip biases use foot contact to turn the body; no simulator-state teleportation.
        action[[0,2,4,6]] -= np.clip(self.steering_gain*error,-self.max_steering,self.max_steering)
        return np.clip(action,-1,1).astype(np.float32)
