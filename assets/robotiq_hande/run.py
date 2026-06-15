import time

import mujoco as mj
import mujoco.viewer
import numpy as np

model = mj.MjModel.from_xml_path(
    "/home/vims/git/mujoco_menagerie_hande/robotiq_hande/scene.xml"
)
data = mj.MjData(model)

obj_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "object")
hande_root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hande")
act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
ten_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_TENDON, "split")

open_ctrl = 255.0
close_ctrl = 255.0
open_seconds = 0.5
open_steps = max(1, int(open_seconds / model.opt.timestep))
step_count = 0


def is_descendant(body_id, root_id):
    while body_id != -1:
        if body_id == root_id:
            return True
        body_id = model.body_parentid[body_id]
    return False


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # if act_id != -1:
        #     data.ctrl[act_id] = open_ctrl if step_count < open_steps else close_ctrl

        mj.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        clamp = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if obj_body != -1:
                contact_ok = (b1 == obj_body and is_descendant(b2, hande_root)) or (
                    b2 == obj_body and is_descendant(b1, hande_root)
                )
            else:
                # Fallback: any contact between the hande subtree and a non-world body.
                b1_in = is_descendant(b1, hande_root)
                b2_in = is_descendant(b2, hande_root)
                contact_ok = (b1_in != b2_in) and (b1 != 0 and b2 != 0)
            if contact_ok:
                f = np.zeros(6, dtype=np.float64)
                mj.mj_contactForce(model, data, i, f)
                clamp += abs(f[0])  # normal component magnitude
        if ten_id != -1 and hasattr(data, "ten_force"):
            tendon_force = data.ten_force[ten_id]
        else:
            # Fallback for older MuJoCo Python bindings.
            tendon_force = data.actuator_force[act_id] if act_id != -1 else 0.0
        print("clamp N:", clamp, "tendon/actuator N:", tendon_force)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # input()

        step_count += 1
