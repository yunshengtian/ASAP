import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import base64
import json

import trimesh

from assets.load import load_assembly

assembly_name = "beam_assembly"
assembly_obj_dir = "~/ASAP/assets/beam_assembly/original"


def export_model(meshes):
    """
    Export assembly mesh to GLB format
    """

    # get a scene object containing the meshes
    scene = trimesh.Scene(meshes)
    scene.camera

    data = scene.export(file_type="glb")
    encoded = base64.b64encode(data).decode("utf-8")

    return encoded

if __name__ == "__main__":

    assembly = load_assembly(assembly_obj_dir)

    meshes = [part["mesh"] for part in assembly.values()]
    names = [part["name"] for part in assembly.values()]
    modelGLB = export_model(meshes)

    assembly_data = {
        "assembly-id": assembly_name,
        "parts": names,
        "model-data": modelGLB,
    }

    output_path = os.path.join(assembly_obj_dir, "viz_data.json")
    with open(output_path, "w") as f:
        json.dump(assembly_data, f)

