const std = @import("std");

pub fn write_image(image: *const [3][]f32, image_width: u32, image_height: u32) !void {
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
