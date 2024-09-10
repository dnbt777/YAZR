// structs probbaly need to match exactly what is in cuda
pub const Sphere = extern struct {
    center: [3]f32,
    radius: f32,
    material_id: u16,
};

pub const Material = extern struct {
    material_type: u16,
    albedo: [3]f32,
}; // for now just put every single property in here

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

// idea
// no triangles
// fucking 0 triangles
// all the geometry will be spheres
// sections of spheres
// i can make a square with 3 floats instead of 4!!!!!! or a rectangle, or anything
// i can make curved surfaces with only 3 floats
// i can make a triangle too out of a sphere
// basically parameterize the sphere right
// wait.. fuck i need rotation too... nooo
// hmm
// ok
// how many points do i need to define an ellipse
// in 2d its like.. 5 points
// fug
// lat0-lat1, long0-long1, r, ...
//
//
//
//
// fourier renderer?
// idk
// we just doin circles for now
