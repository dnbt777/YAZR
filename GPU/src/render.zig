const phys = @import("physics.zig");

// ok so, if we're operating on a matrix, why is M a pointer to a single f32?
// the reason is bc cuda uses the first pointer to find the start of the matrix
// then it uses width and height to move over in memory to the other parts
// hence why we call fillMat(&r[0]..  rather than fillMat(&r..
// we dont pass in the whole R channel, just a pointer to its first value
pub extern "c" fn initScene(
    width: u16,
    height: u16,
    pixel_delta_u0: f32, // degenerate way to do it but.. whatever
    pixel_delta_u1: f32,
    pixel_delta_u2: f32,
    pixel_delta_v0: f32,
    pixel_delta_v1: f32,
    pixel_delta_v2: f32,
    pixel00_loc0: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc1: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc2: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    origin0: f32, // not sure if this should be a slice...
    origin1: f32, // not sure if this should be a slice...
    origin2: f32, // not sure if this should be a slice...
    samples: u16,
    level_geometry_host: *phys.Sphere, // pass in level_geometry_host[0]. in cuda the signature is phys.Sphere* indicating array
    level_objects_host: *phys.Sphere,
    static_obj_count: u16,
    dynamic_obj_count: u16,
    materials_host: *phys.Material,
    material_count: u16,
) u16; // returns failure code or w/e

pub extern "c" fn render_scene(
    width: u16,
    height: u16,
    pixel_delta_u0: f32, // degenerate way to do it but.. whatever
    pixel_delta_u1: f32,
    pixel_delta_u2: f32,
    pixel_delta_v0: f32,
    pixel_delta_v1: f32,
    pixel_delta_v2: f32,
    pixel00_loc0: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc1: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc2: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    origin0: f32, // not sure if this should be a slice...
    origin1: f32, // not sure if this should be a slice...
    origin2: f32, // not sure if this should be a slice...
    samples: u16,
    level_objects_host: *phys.Sphere,
    dynamic_obj_count: u16,
) void;
