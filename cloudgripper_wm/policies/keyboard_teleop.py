"""Keyboard teleoperation for the CloudGripper MuJoCo sim.

One keypress produces one action of fixed size, and the simulation blocks
until you press something — so an episode advances only as fast as you drive
it.

Keys are read from the *terminal* in raw mode rather than from the MuJoCo
viewer window. Two reasons: the viewer reserves w/s/e/t/r/f/q/g for its own
display toggles (wireframe, shadows, reflections), so anything it sees changes
the rendering instead of the arm; and its key_callback fires from the render
thread, which cannot block the policy.

This means the terminal must have focus while you drive. The viewer still
updates each step, so you watch one window and type into the other.
"""

import sys
import termios
import tty

import numpy as np

from stable_worldmodel.policy import BasePolicy


class KeyboardTeleopPolicy(BasePolicy):
    """Blocking teleoperation: one keypress, one action, no repeats.

    Args:
        step_size: magnitude of each action, in normalised joint units.
            Note the env ignores commands whose accumulated target is within
            step_threshold (0.01 by default) of the current position, so a
            step_size at or below that may actuate only intermittently.
        max_delta: action-space bound; actions are clipped to it.
    """

    KEYS = {
        'w': (0, +1), 's': (0, -1),      # Rail_joint    — x
        'd': (1, +1), 'a': (1, -1),      # Slider_joint  — y
        'e': (2, +1), 'q': (2, -1),      # Linear_joint  — z (1.0 is lowest)
        'r': (3, +1), 'f': (3, -1),      # Rotation_joint
        't': (4, +1), 'g': (4, -1),      # RightSpur_joint — gripper
    }

    def __init__(self, step_size=0.02, max_delta=0.05, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.step_size = float(step_size)
        self.max_delta = float(max_delta)
        self._viewer_env = None
        self._help_shown = False

    def set_env(self, envs):
        """Called by swm.World. Keep the unwrapped env so we can sync the viewer.

        EnvPool holds the wrapped envs; .unwrapped reaches the
        CloudgripperMuJoCoEnv that owns the passive viewer.
        """
        super().set_env(envs)
        try:
            self._viewer_env = envs.envs[0].unwrapped
        except (AttributeError, IndexError):
            self._viewer_env = None

    def _read_key(self):
        """Block until one key is pressed, then discard anything buffered.

        tty.setraw makes the terminal deliver each character immediately
        rather than waiting for a newline. The tcflush afterwards throws away
        auto-repeat characters that piled up while a key was held, so holding
        a key does not queue a burst of actions.
        """
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            termios.tcflush(fd, termios.TCIFLUSH)
        return ch

    def get_action(self, obs, **kwargs):
        """One keypress, one action. Blocks, so the sim waits for you."""
        if not self._help_shown:
            print("\n  w/s = x    a/d = y    q/e = z"
                  "    f/r = rotate    g/t = gripper"
                  "\n  space = hold still    Ctrl-C = stop"
                  "\n  (click the TERMINAL window, not the viewer)\n",
                  flush=True)
            self._help_shown = True

        if self._viewer_env is not None:
            try:
                self._viewer_env.sync_passive_viewer()
            except Exception:
                pass

        ch = self._read_key()

        if ch == '\x03':                      # Ctrl-C: raw mode swallows the
            raise KeyboardInterrupt           # usual interrupt, so handle it

        action = np.zeros(5, dtype=np.float32)
        key = ch.lower()
        if key in self.KEYS:
            axis, sign = self.KEYS[key]
            action[axis] = sign * self.step_size

        action = np.clip(action, -self.max_delta, self.max_delta)
        return action[None, :].astype(np.float32)

    def reset(self):
        """Called between episodes."""
        self._help_shown = False