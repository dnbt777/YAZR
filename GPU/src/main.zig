const std = @import("std");

fn write_image(image: *const [3][]f32, image_width: u32, image_height: u32) !void {
    const file = try std.fs.cwd().createFile(
        "img.ppm",
        .{ .read = true },
    );
    defer file.close();

    // write header
    var header_buffer: [128]u8 = undefined;
    const formatted_string = try std.fmt.bufPrint(&header_buffer, "P3\n{} {}\n255\n", .{ image_width, image_height });

    try file.writeAll(formatted_string);

    // write pixels in image
    for (0..image_width) |wi| {
        const w: u16 = @intCast(wi);
        for (0..image_height) |hi| {
            const h: u16 = @intCast(hi);

            const pixel_index = h * image_width + w;
            const r: f32 = image[0][pixel_index];
            const g: f32 = image[1][pixel_index];
            const b: f32 = image[2][pixel_index];

            var pix_buffer: [128]u8 = undefined;
            const formatted_pix_string = try std.fmt.bufPrint(&pix_buffer, "{d} {d} {d}\n", .{
                @as(u8, @intFromFloat(r)),
                @as(u8, @intFromFloat(g)),
                @as(u8, @intFromFloat(b)),
            });
            try file.writeAll(formatted_pix_string);
        }
    }
}

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

    try write_image(&image, image_width, image_height);
}
