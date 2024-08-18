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
    var camera: engine.Camera = engine.final_render_camera();

    // dontask me why this is an arena
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var world_list: hittable.HittableList = hittable.HittableList{ .objects = std.ArrayList(hittable.Hittable).init(allocator) };

    const ground_material: material.Material = material.Material{
        .mat_id = 1,
        .albedo = utils.vec3(0.5, 0.5, 0.5),
        .fuzz = undefined,
        .refraction_index = undefined,
    };
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
        .mat = &ground_material,
        .center = utils.vec3(0, -1000, 0),
        .radius = 1000,
    } });

    const sphere_count_sqrt: usize = 22;
    const sphere_count: usize = sphere_count_sqrt * sphere_count_sqrt;
    var sphere_materials: [sphere_count]material.Material = undefined;

    for (0..sphere_count_sqrt) |ai| {
        const a: i8 = @as(i8, @intCast(ai)) - 11;
        for (0..sphere_count_sqrt) |bi| {
            const b: i8 = @as(i8, @intCast(bi)) - 11;
            const sphere_index = ai * sphere_count_sqrt + bi;
            const choose_mat: f32 = utils.random_double();
            std.debug.print("{} => ", .{choose_mat});
            const center: @Vector(3, f32) = utils.vec3(@as(f32, @floatFromInt(a)) + 0.9 * utils.random_double(), 0.2, @as(f32, @floatFromInt(b)) + 0.9 * utils.random_double());

            const vx = center - utils.vec3(4, 0.2, 0);
            if (@sqrt(utils.dot(vx, vx)) > 0.9) {
                if (choose_mat < 0.8) {
                    // diffuse
                    std.debug.print("diffuse\n", .{});
                    const albedo = utils.random_vec3() * utils.random_vec3();
                    std.debug.print("albedo={}\n", .{albedo});
                    sphere_materials[sphere_index] = material.Material{
                        .mat_id = 1,
                        .albedo = albedo,
                        .fuzz = undefined,
                        .refraction_index = undefined,
                    };
                    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
                        .mat = &sphere_materials[sphere_index],
                        .center = center,
                        .radius = 0.2,
                    } });
                } else if (choose_mat < 0.95) {
                    // metal
                    std.debug.print("metal\n", .{});
                    const albedo = utils.random_vec3_range(0.5, 1);
                    std.debug.print("albedo={}\n", .{albedo});
                    sphere_materials[sphere_index] = material.Material{
                        .mat_id = 2,
                        .albedo = albedo,
                        .fuzz = utils.random_double_range(0, 0.01),
                        .refraction_index = undefined,
                    };
                    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
                        .mat = &sphere_materials[sphere_index],
                        .center = center,
                        .radius = 0.2,
                    } });
                } else {
                    // glass
                    std.debug.print("glass\n", .{});
                    sphere_materials[sphere_index] = material.Material{
                        .mat_id = 3,
                        .albedo = undefined,
                        .fuzz = undefined,
                        .refraction_index = 1.5,
                    };
                    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{
                        .mat = &sphere_materials[sphere_index],
                        .center = center,
                        .radius = 0.2,
                    } });
                }
            }
        }
    }

    const material1 = material.Material{
        .mat_id = 1,
        .albedo = utils.vec3(0.4, 0.2, 0.1),
        .fuzz = undefined,
        .refraction_index = undefined,
    };
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{ .mat = &material1, .radius = 1.0, .center = utils.vec3(-4, 1, 0) } });

    const material2 = material.Material{
        .mat_id = 2,
        .albedo = utils.vec3(0.7, 0.6, 0.5),
        .fuzz = 0.0,
        .refraction_index = undefined,
    };
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{ .mat = &material2, .radius = 1.0, .center = utils.vec3(4, 1, 0) } });

    const material3 = material.Material{
        .mat_id = 3,
        .albedo = undefined,
        .fuzz = undefined,
        .refraction_index = 1.5,
    };
    try world_list.add(hittable.Hittable{ .sphere = hittable.Sphere{ .mat = &material3, .radius = 1.0, .center = utils.vec3(0, 1, 0) } });

    const world: hittable.Hittable = hittable.Hittable{ .hittable_list = world_list };
    try camera.render(world);
}
