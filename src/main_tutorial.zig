const std = @import("std");
const fs = std.fs;
const meta = std.meta;
const imgwriter = @import("imgwriter.zig");
const inf = std.math.inf(f32);
const interval = @import("interval.zig");
const hittable = @import("hittable.zig");
const utils = @import("utils.zig");
const rt = @import("rt.zig");
const engine = @import("engine.zig");
const material = @import("material.zig");

// turn this into non euclidean risk of rain wasm
pub fn main() !void {
    var camera: engine.Camera = engine.default_camera();

    // dontask me why this is an arena
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var world_list: hittable.HittableList = hittable.HittableList{ .objects = std.ArrayList(hittable.Hittable).init(allocator) };

    const material_right = material.Material{
        .mat_id = 2,
        .albedo = utils.vec3(1, 0, 0),
        .fuzz = 1.0,
        .refraction_index = undefined,
    };
    const material_left = material.Material{
        .mat_id = 3,
        .albedo = undefined,
        .fuzz = undefined,
        .refraction_index = 1.50,
    };
    const ground = material.Material{
        .mat_id = 1,
        .albedo = utils.vec3(0.8, 0.8, 0.0),
        .fuzz = undefined,
        .refraction_index = undefined,
    };
    const center = material.Material{
        .mat_id = 1,
        .albedo = utils.vec3(0.1, 0.2, 0.5),
        .fuzz = undefined,
        .refraction_index = undefined,
    };
    const bubble = material.Material{
        .mat_id = 3,
        .albedo = undefined,
        .fuzz = undefined,
        .refraction_index = (1.00 / 1.50),
    };

    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .center = utils.vec3(-1, 0, -1),
        .radius = 0.5,
        .mat = &material_left,
    } });
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .center = utils.vec3(1, 0, -1),
        .radius = 0.5,
        .mat = &material_right,
    } });
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .center = utils.vec3(0, -100.5, -1),
        .radius = 100,
        .mat = &ground,
    } });
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .center = utils.vec3(0, 0, -1.2),
        .radius = 0.5,
        .mat = &center,
    } });
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .center = utils.vec3(-1, 0, -1),
        .radius = 0.4,
        .mat = &bubble,
    } });

    const world: hittable.Hittable = hittable.Hittable{ .hittable_list = world_list };

    try camera.render(world);
}
