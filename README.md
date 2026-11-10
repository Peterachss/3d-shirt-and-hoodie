# 3D Blender Shirt & Hoodie Models

Parametric quad-topology 3D crewneck T-shirt and pullover hoodie models built automatically in Blender via GitHub Actions.

## Files

- `shirt_and_hoodie.blend`: Ready-to-open Blender scene with `T-Shirt`, `Hoodie`, and `Studio` collections, Subdivision Surface modifiers, Principled BSDF fabric shaders with micro-weave bump, and 3-point lighting.
- `blender_studio_render.png`: Cycles render produced directly by Blender on GitHub Actions.
- `tshirt.obj` / `tshirt.glb`: Crewneck T-shirt with thickened shell, set-in sleeves, ribbed collar, and hem.
- `hoodie.obj` / `hoodie.glb`: Pullover hoodie with dropped-shoulder sleeves, ribbed waistband and cuffs, domed double-layer hood, kangaroo pocket, and drawstrings with metal aglets.
- `shirt_and_hoodie.obj` / `shirt_and_hoodie.glb`: Combined side-by-side scene.
- `garment_builder.py`: Parametric quad-mesh generator.
- `build_garments_blender.py`: Blender `bpy` scene and shader setup script.
