// structs probbaly need to match exactly what is in cuda
pub const Sphere = extern struct {
    center: [3]f32,
    radius: f32,
};

pub const PhysicsProperties = extern struct {
    velocity: @Vector(3, f32), // vector if the CPU works with it, [3]f32 if its sent to GPU.
    mass: f32,
    acceleration: @Vector(3, f32), // in this case the physics is done CPU sided
    position: @Vector(3, f32),
};

pub const PhysicsObjects = extern struct {
    geometries: [256]Sphere, // only spheres for now. this is what gets sent to the renderer
    physical_properties: [256]PhysicsProperties,
};
