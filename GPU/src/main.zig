const std = @import("std");
const display = @import("display.zig");

fn fillMatrix(matrix: *[]f32, c: f32) void {
    for (matrix.*) |*value| {
        value.* = c;
    }
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const N = 100; // square for now
    const image_height = N;
    const image_width = N;

    // init image w r g and b channels
    var image: [3][]f32 = .{
        try allocator.alloc(f32, image_height * image_width),
        try allocator.alloc(f32, image_height * image_width),
        try allocator.alloc(f32, image_height * image_width),
    };

    fillMatrix(&image[0], 254.0);
    fillMatrix(&image[1], 100.0);
    fillMatrix(&image[2], 100.0);

    // Writes to ppm. alternate in future: write to some screen buffer or something
    try display.write_image(&image, image_width, image_height);
}
