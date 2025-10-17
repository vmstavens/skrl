import queue
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock, Thread
from typing import Callable, List

import mujoco as mj
import mujoco.viewer


class SimSync:
    def __init__(self):
        self.can_step = Lock()
        self.did_step = Lock()
        # lock at init
        self.can_step.acquire()
        self.did_step.acquire()

    def step(self):
        self.can_step.release()  # allows step to progress
        self.did_step.acquire()  # waits for step to be done


def sleep(duration, sim_sync: SimSync, model: mj.MjModel):
    n_steps = int(round(duration / model.opt.timestep))
    for _ in range(n_steps):
        sim_sync.step()


class BaseSim(ABC):
    """
    Abstract base class for MuJoCo robot simulations.

    This class serves as the foundational framework for simulating robots in MuJoCo. It defines
    key properties and methods that must be implemented in subclasses, including access to
    the simulation's dynamic data (`MjData`), the static model (`MjModel`), and control mechanisms.

    Child classes are responsible for implementing the control logic (`control_loop`) and handling
    user inputs via the keyboard (`keyboard_callback`). The `run` method provides a basic structure
    for running the main simulation loop, handling viewer synchronization, and processing user inputs
    in real-time.

    This base class includes functionality to issue warnings if important methods, such as the control
    loop or keyboard callback, are not implemented in a subclass.
    """

    def __init__(self) -> None:
        self._control_loop_warning_issued = False
        self._keyboard_callback_warning_issued = False
        self.threads: List[Callable] = []
        self._scene = None

    @property
    @abstractmethod
    def data(self) -> mj.MjData:
        """
        Access the current simulation data.

        This property provides access to an instance of the `MjData` class, which contains the dynamic
        simulation state. This includes quantities such as joint positions, velocities,
        actuator forces, and sensory information. The `MjData` object is updated at each simulation step
        and can be used to inspect the real-time state of said simulation.

        Returns
        -------
        mj.MjData
            An object representing the current dynamic state of the simulation.
        """
        raise NotImplementedError("property 'data' must be implemented in simulation.")

    @property
    @abstractmethod
    def model(self) -> mj.MjModel:
        """
        Access the model of the MuJoCo simulation.

        This property returns an instance of the `MjModel` class, which describes the physical and
        mechanical properties of the simulation. The `MjModel` object contains static information about the
        simulation such as its kinematic trees, inertial properties, joint and actuator definitions, and geometry
        configurations. It is used to define the system's structure and behavior within the simulation.

        Returns
        -------
        mj.MjModel
            An object representing the static model of the system and overall MuJoCo simulation.
        """
        raise NotImplementedError("property 'model' must be implemented in robot.")

    @property
    def scene(self) -> mj.MjvScene:
        assert self._scene is not None, (
            "scene attribute is None, likely due to no scene being defined in viewer loop. "
            "Ensure this attribute is not accessed before viewer."
        )
        return self._scene

    def control_loop(self) -> None:
        """
        Main control loop for the robot simulation.

        This method is intended to be overridden in the child class to implement the robot's control
        logic. The control loop is typically responsible for computing control signals based on the
        current state of the simulation (e.g., sensor data) and applying these signals to the robot's
        actuators. The frequency of the control loop matches the simulation step rate.

        If this method is not implemented in the child class, a warning will be issued.

        Raises
        ------
        UserWarning
            If the method is not implemented in the child class.
        """
        if not self._control_loop_warning_issued:
            warnings.warn(
                "Method 'control_loop' is not implemented in the sim child class.",
                UserWarning,
            )
            self._control_loop_warning_issued = True
        return

    def keyboard_callback(self, key: int) -> None:
        """
        Handle keyboard inputs during the simulation.

        This method is intended to be overridden in the child class to define how the simulation responds
        to specific keyboard inputs. The `key` parameter corresponds to the key pressed by the user,
        allowing for custom behavior (e.g., sending trajectories to robots' task queues, logging data
        or start/stopping processes).

        If this method is not implemented in the child class, a warning will be issued.

        Parameters
        ----------
        key : int
            The key pressed by the user represented by a key code.

        Raises
        ------
        UserWarning
            If the method is not implemented in the child class.
        """
        if not self._keyboard_callback_warning_issued:
            warnings.warn(
                "Method 'keyboard_callback' is not implemented in the sim child class.",
                UserWarning,
            )
            self._keyboard_callback_warning_issued = True
        return

    def run(
        self,
        headless: bool = False,
        limit_cycle_time: bool = True,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        dyn: bool = True,
    ) -> None:
        """
        Run the main simulation loop with or without the MuJoCo viewer.

        This method starts the simulation loop, either with the MuJoCo viewer or in headless mode,
        depending on the `headless` argument. In non-headless mode, the viewer is launched to manage
        keyboard events, synchronize with the simulation, and control the timing of each simulation step.
        The camera view is rendered if enabled. In headless mode, the simulation runs in the background
        without any visual output, stepping through each simulation step while performing the control loop.

        Parameters
        ----------
        headless : bool, optional
            If True, the simulation runs without rendering or a viewer (default is False).

        Returns
        -------
        None
        """

        # create SimSync objects for each task
        sim_syncs = [SimSync() for _ in range(len(self.threads))]

        threads = [
            Thread(
                target=tgt,
                args=(ss,),
            )
            for tgt, ss in zip(self.threads, sim_syncs)
        ]

        for i, t in enumerate(threads):
            t.start()

        # run the simulation headless
        if headless:
            print("Running MuJoCo Headless...")
            while True:
                # mj.mj_step(self.model, self.data)
                # if a task is done delete the
                for t in threads:
                    if not t.is_alive():
                        print(
                            f"[{Path(__file__).stem}]: Process '{t.name}' terminated successfully..."
                        )
                        del threads[i], sim_syncs[i]

                # wait for all sim synchs to acquire their lock i.e.
                # all threads have successfully completed their last step
                for i, ss in enumerate(sim_syncs):
                    ss.can_step.acquire()

                # step simulation one time step
                mj.mj_step(self.model, self.data)

                # we now wait for all the threads to reach the point
                # where they have all stepped and released their
                # "did_step" lock
                for ss in sim_syncs:
                    ss.did_step.release()
                self.control_loop()

        # in order to enable camera rendering in main thread, queue the key events
        key_queue = queue.Queue()

        with mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            key_callback=lambda key: key_queue.put(key),
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        ) as viewer:
            # set gui camera to the specified in the model
            viewer.cam.azimuth = self.model.vis.global_.azimuth
            viewer.cam.elevation = self.model.vis.global_.elevation
            viewer.cam.lookat = self.model.stat.center
            viewer.cam.distance = self.model.stat.extent

            self._scene = viewer.user_scn

            while viewer.is_running():
                step_start = time.time()

                while not key_queue.empty():
                    self.keyboard_callback(key_queue.get())

                # if a task is done delete the
                for t in threads:
                    if not t.is_alive():
                        print(
                            f"[{Path(__file__).stem}]: Process '{t.name}' terminated successfully..."
                        )
                        del threads[i], sim_syncs[i]

                # wait for all sim synchs to acquire their lock i.e.
                # all threads have successfully completed their last step
                for i, ss in enumerate(sim_syncs):
                    ss.can_step.acquire()

                if dyn:
                    # step simulation one time step
                    mj.mj_step(self.model, self.data)
                else:
                    # The minimal function call required to get updated frame transforms is
                    # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
                    mj.mj_kinematics(self.model, self.data)
                    mj.mj_comPos(self.model, self.data)
                    if self.model.neq > 0:
                        mj.mj_makeConstraint(self.model, self.data)

                # we now wait for all the threads to reach the point
                # where they have all stepped and released their
                # "did_step" lock
                for ss in sim_syncs:
                    ss.did_step.release()

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                if limit_cycle_time:
                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
