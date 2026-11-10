"""
Blender Scene Builder for the 3D T-Shirt and Pullover Hoodie
Run inside Blender (Scripting tab -> Run Script) or via CLI:
    blender --background --python build_garments_blender.py -- --render
Builds Collections ('T-Shirt', 'Hoodie', 'Studio'), enables Shade Smooth +
Subdivision Surface, configures Principled BSDF fabric shaders with micro-weave
bump, sets 3-point studio lighting + camera, saves 'shirt_and_hoodie.blend',
and optionally renders 'blender_studio_render.png'.
"""

from pathlib import Path
import sys
import bpy
from mathutils import Vector


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_or_create_collection(name: str, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    if parent is None:
        bpy.context.scene.collection.children.link(col)
    else:
        parent.children.link(col)
    return col


def import_obj_file(obj_path: Path, target_collection, x_offset: float) -> list:
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="NEGATIVE_Z", up_axis="Y")
    else:
        bpy.ops.import_scene.obj(filepath=str(obj_path), axis_forward="-Z", axis_up="Y")
    imported = [obj for obj in bpy.data.objects if obj not in before]

    for obj in imported:
        for col in list(obj.users_collection):
            col.objects.unlink(obj)
        target_collection.objects.link(obj)
        obj.location.x += x_offset

        if obj.type == "MESH":
            for poly in obj.data.polygons:
                poly.use_smooth = True

            is_cord = "Drawstring" in obj.name or "Aglet" in obj.name
            sub = obj.modifiers.new(name="Subdivision", type="SUBSURF")
            sub.levels = 1
            sub.render_levels = 1 if is_cord else 2

    return imported


def enhance_fabric_materials() -> None:
    """Upgrade imported materials with procedural cotton/fleece weave bump and cloth sheen."""
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue

        is_metal = "metal" in mat.name.lower()
        if not is_metal:
            for sock_name, val in [("Sheen Weight", 0.45), ("Sheen", 0.45), ("Sheen Roughness", 0.6)]:
                if sock_name in bsdf.inputs:
                    bsdf.inputs[sock_name].default_value = val

            tex_coord = nt.nodes.new("ShaderNodeTexCoord")
            tex_coord.location = (-680, -220)

            wave_u = nt.nodes.new("ShaderNodeTexWave")
            wave_u.location = (-460, -160)
            wave_u.wave_type = "BANDS"
            wave_u.bands_direction = "X"
            if "Scale" in wave_u.inputs:
                wave_u.inputs["Scale"].default_value = 180.0 if "rib" in mat.name.lower() else 260.0
            if "Distortion" in wave_u.inputs:
                wave_u.inputs["Distortion"].default_value = 1.2

            bump = nt.nodes.new("ShaderNodeBump")
            bump.location = (-220, -200)
            if "Strength" in bump.inputs:
                bump.inputs["Strength"].default_value = 0.08 if "rib" in mat.name.lower() else 0.045
            if "Distance" in bump.inputs:
                bump.inputs["Distance"].default_value = 0.002

            nt.links.new(tex_coord.outputs["UV"], wave_u.inputs["Vector"])
            nt.links.new(wave_u.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def setup_studio(studio_col) -> None:
    # Build matte floor directly via mesh.from_pydata (no active-context ops)
    floor_mesh = bpy.data.meshes.new("Studio_Floor_Mesh")
    floor_mesh.from_pydata(
        [(-4.0, -4.0, -0.002), (4.0, -4.0, -0.002), (4.0, 4.0, -0.002), (-4.0, 4.0, -0.002)],
        [],
        [(0, 1, 2, 3)],
    )
    floor_mesh.update()
    floor = bpy.data.objects.new("Studio_Floor", floor_mesh)
    studio_col.objects.link(floor)

    floor_mat = bpy.data.materials.new("Studio_Floor_Mat")
    floor_mat.use_nodes = True
    bsdf = next((n for n in floor_mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.88, 0.87, 0.84, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.92
    floor.data.materials.append(floor_mat)

    def add_area_light(name: str, loc, energy: float, size: float, target=(0.0, 0.0, 0.45)):
        ldata = bpy.data.lights.new(name=name, type="AREA")
        ldata.energy = energy
        ldata.size = size
        lobj = bpy.data.objects.new(name=name, object_data=ldata)
        lobj.location = loc
        studio_col.objects.link(lobj)
        direction = Vector(target) - Vector(loc)
        lobj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    add_area_light("Key_Light", (1.8, -2.2, 2.0), energy=150.0, size=1.8)
    add_area_light("Fill_Light", (-2.2, -1.8, 1.4), energy=75.0, size=2.4)
    add_area_light("Rim_Light", (0.2, 2.3, 1.9), energy=110.0, size=1.6)

    cam_data = bpy.data.cameras.new("Studio_Camera")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("Studio_Camera", cam_data)
    cam_obj.location = (0.0, -2.45, 0.56)
    cam_obj.rotation_euler = (Vector((0.0, 0.0, 0.48)) - Vector(cam_obj.location)).to_track_quat("-Z", "Y").to_euler()
    studio_col.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    world = bpy.data.worlds.new("Studio_World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.91, 0.90, 0.88, 1.0)
        bg.inputs["Strength"].default_value = 0.50

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800
    scene.render.image_settings.file_format = "PNG"


def main() -> None:
    folder = Path(__file__).resolve().parent
    tshirt_obj = folder / "tshirt.obj"
    hoodie_obj = folder / "hoodie.obj"

    if not tshirt_obj.exists() or not hoodie_obj.exists():
        sys.path.insert(0, str(folder))
        import garment_builder
        garment_builder.main()

    clear_scene()
    tshirt_col = get_or_create_collection("T-Shirt")
    hoodie_col = get_or_create_collection("Hoodie")
    studio_col = get_or_create_collection("Studio")

    import_obj_file(tshirt_obj, tshirt_col, x_offset=-0.56)
    import_obj_file(hoodie_obj, hoodie_col, x_offset=0.56)
    enhance_fabric_materials()
    setup_studio(studio_col)

    blend_out = folder / "shirt_and_hoodie.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out), compress=True)
    print(f"Saved Blender scene: {blend_out}")

    if "--render" in sys.argv:
        render_out = folder / "blender_studio_render.png"
        bpy.context.scene.render.filepath = str(render_out)
        bpy.ops.render.render(write_still=True)
        print(f"Saved Cycles studio render: {render_out}")


if __name__ == "__main__":
    main()
