# Third-party sources

The PPO architecture, orthogonal initialization, clipped policy/value objectives,
GAE, optimizer settings, and TensorBoard conventions in training/ppo.py are
adapted from CleanRL's ppo.py and ppo_continuous_action.py:

- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

This project adds decentralized shared-policy rollouts, per-agent inactive masks,
explicit time-limit bootstrapping, and a tanh Gaussian with its Jacobian correction.
It does not use Stable-Baselines3 or RLlib.

MIT License

Copyright (c) 2019 CleanRL developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Gymnasium Ant model

`envs/ant_template.xml` is copied from the installed Gymnasium distribution;
`ant_world.xml` is generated from it by `ant_model.py`. Ant geometry, joints, and
motor parameters originate in Gymnasium's standard Ant model:
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/mujoco/assets/ant.xml
The generated world enables inter-robot collisions and prefixes all names.

## Pretrained CleanRL Ant locomotion policy

The optional five-action 3D controller uses tensor weights from
`sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2`, pinned to revision
`b88f77df64955077718a67c826f2dc290dfbe4e9`. The repository reports that the
policy was trained with CleanRL PPO on Gymnasium Ant-v4. The download script
retrieves only the checkpoint tensor file and records its SHA-256; downloaded
Python code is not executed. See the upstream model card and its license terms:
https://huggingface.co/sdpkjc/Ant-v4-ppo_fix_continuous_action-seed2

The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

