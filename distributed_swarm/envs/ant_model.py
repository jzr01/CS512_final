"""Combine the Gymnasium Ant body and motors into one collision-enabled world."""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET


def build_ant_world(num_agents=5):
    root = ET.parse(Path(__file__).with_name("ant_template.xml")).getroot()
    root.set("model", "swarm_ant")
    custom = root.find("custom")
    if custom is not None:
        root.remove(custom)
    root.find("default/geom").set("conaffinity", "1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="800", offheight="600")
    world = root.find("worldbody")
    template = world.find("body")
    world.remove(template)
    motors = list(root.find("actuator"))
    actuators = root.find("actuator")
    actuators.clear()
    for i in range(num_agents):
        body = copy.deepcopy(template)
        body.set("pos", f"{(i % 3) * 3.0 + 2} {(i // 3) * 3.0 + 2} 0.75")
        # Give every body, geom, joint, camera, and motor a distinct name.
        for k, element in enumerate(body.iter()):
            element.set("name", f"ant{i}_" + element.get("name", f"{element.tag}{k}"))
        world.append(body)
        for k, original in enumerate(motors):
            motor = copy.deepcopy(original)
            motor.set("name", f"ant{i}_motor{k}")
            motor.set("joint", f"ant{i}_" + motor.get("joint"))
            actuators.append(motor)
    ET.SubElement(world, "site", name="goal", pos="16 16 0.02", size="2 2 .01",
                  type="cylinder", rgba="0.1 0.9 0.3 0.3")
    return ET.tostring(root, encoding="unicode")
