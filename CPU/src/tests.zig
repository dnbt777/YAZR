const std = @import("std");
const utils = @import("utils.zig");
const hittable = @import("hittable.zig");
const rt = @import("rt.zig");
const engine = @import("engine.zig");
const imgwriter = @import("imgwriter.zig");

const expect = std.testing.expect;
test "utils_test" {
    try expect(utils.random_double() <= 1);
    try expect(utils.random_double() >= 0);
    const rand1 = utils.random_double();
    const rand2 = utils.random_double();
    //try expect(rand1 != rand2); // if this fails thatd be hilarious lol
    std.log.warn("\ntwo random nums: {} {}\n", .{ rand1, rand2 });
}

test "hittable_tests" {}

test "raytracer_tests" {}

// camera stuff and whatnot
test "engine tests" {
    var camera: engine.Camera = undefined;
    _ = &camera.initialize();
    std.log.warn("\ncamera.pixel_samples_scale={}\n", .{camera.pixel_samples_scale});
}

// actually writing out to images
test "imgwriter tests" {}
