"""Convert the official S10 + Piper link meshes for the browser viewer."""
import json
import os
from pathlib import Path

import trimesh

root = Path(__file__).resolve().parents[1]
sdk = Path(os.environ.get("S10_SDK_ROOT", root / "spikes/s10-sdk2/sim"))
meshes = sdk / "S10_description/S10_Piper/meshes"
dest = root / "frontend/public/models"
links = json.loads((dest / "s10-piper-rig.json").read_text())
for name in sorted({mesh for link in links for mesh in link["meshes"]}):
    source, = meshes.rglob(f"{name}.STL")
    trimesh.load_mesh(source).export(dest / f"{name}.glb")
print("S10 + Piper browser models ready.")
