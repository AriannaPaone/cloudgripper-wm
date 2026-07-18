import gymnasium as gym
import mujoco

class Cloudgripper(gym.Env):
    """Wrapper for a basic Mujoco Cloudgripper cell.
    
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10,
        'render_fps': 10,
    }

    def __init__(
        self,
        block_cog=None,
        render_action=False,
        resolution=224,
        with_target=True,
        render_mode : str | None ='rgb_array',
        relative=True,
        init_value=None,
        width: int = 200,
        height: int = 200,
    ):
        
        self.render_mode : str | None = render_mode
        self._path_to_xml = "environments/mj_cloudgripper/cloudgripper_cell.xml"
        
        # mujoco related
        self._model: mujoco.MjModel | None     = None 
        self._data: mujoco.MjData | None       = None
        self._renderer: mujoco.Renderer | None = None
        self._camera = mujoco.MjvCamera()
        self._render_height = height
        self._render_width = width
        pass

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._model = mujoco.MjModel.from_xml_path(self._path_to_xml)
        self._data = mujoco.MjData(self._model)

        pass

    
    def step(self, action):
        pass

    def render(
        self,
        camera : str | None = None,
        depth : bool = False,
        segmentation : bool = False
    ):
        """Render the current state of the environment."""
        
        camera = 'top' if camera is None else camera
        
        if self._model is None or self._data is None:
            raise ValueError('call `reset` before render.')

        if depth and segmentation:
            raise ValueError('Only one of depth or segmentation can be enabled.')
        
        if self._renderer is None:
            self._init_renderer()
        
        if depth:
            self._renderer.enable_depth_rendering()
        elif segmentation:
            self._renderer.enable_segmentation_rendering()
        else:
            self._renderer.disable_depth_rendering()
            self._renderer.disable_segmentation_rendering()

        img = self.env.render()
        return self._render_frame(self.render_mode)
    
    def close(self):
        if self.window is not None:
            pass
    
    def _render_frame(self, mode : str | None):
        if mode == 'human':
            pass

        pass
    

    def _initialize_renderer(self):
        """Initialize the renderer."""
        
        if self._model is None:
            raise ValueError('Call `reset` before rendering.')
        
        self._renderer = mujoco.Renderer(
            model=self._model,
            height=self._render_height,
            width=self._render_width
        )
        
        mujoco.mjv_defaultFreeCamera(self._model, self._camera)

if __name__ == '__main__':
    cloudgripper = Cloudgripper()
    cloudgripper.reset()