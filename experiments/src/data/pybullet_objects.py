import pybullet as p
from .textures import load_texture

def create_object(shape_type, size_scale, color_rgba, material_specular, texture_type):
    base_radius = 0.5 * size_scale
    base_height = 1.0 * size_scale

    if shape_type == "sphere":
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=base_radius)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=base_radius, rgbaColor=color_rgba)
    elif shape_type in ["box", "cube"]:
        half_extents = [base_radius, base_radius, base_radius]
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color_rgba)
    elif shape_type == "cylinder":
        col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=base_radius, height=base_height)
        vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=base_radius, length=base_height, rgbaColor=color_rgba)
    elif shape_type == "capsule":
        col_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=base_radius, height=base_height)
        vis_shape = p.createVisualShape(p.GEOM_CAPSULE, radius=base_radius, length=base_height, rgbaColor=color_rgba)
    else:
        raise ValueError(f"Unknown shape: {shape_type}")

    obj_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=[0, 0, 0.5 + base_radius]
    )

    if material_specular:
        p.changeVisualShape(obj_id, -1, specularColor=material_specular)

    if texture_type and texture_type != "plain":
        tex_id = load_texture(texture_type)
        if tex_id >= 0:
            p.changeVisualShape(obj_id, -1, textureUniqueId=tex_id)

    return obj_id
