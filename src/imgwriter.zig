const std = @import("std");
const interval = @import("interval.zig");
const utils = @import("utils.zig");

pub fn write_image_header(file: std.fs.File, image_width: u16, image_height: u16) !void {
    var buffer: [128]u8 = undefined;
    const formatted_string = try std.fmt.bufPrint(&buffer, "P3\n{} {}\n255\n", .{ image_width, image_height });

    try file.writeAll(formatted_string);
}

pub fn write_color(file: std.fs.File, color: @Vector(3, f32)) !void {
    const intensity: interval.Interval = interval.Interval{
        .min = 0,
        .max = 0.999,
    };

    const r = utils.linear_to_gamma(color[0]);
    const g = utils.linear_to_gamma(color[1]);
    const b = utils.linear_to_gamma(color[2]);

    const rbyte: u8 = @intFromFloat(255.999 * intensity.clamp(r));
    const gbyte: u8 = @intFromFloat(255.999 * intensity.clamp(g));
    const bbyte: u8 = @intFromFloat(255.999 * intensity.clamp(b));

    var buffer: [128]u8 = undefined;
    const formatted_string = try std.fmt.bufPrint(&buffer, "{} {} {}\n", .{ rbyte, gbyte, bbyte });
    try file.writeAll(formatted_string);
}

pub fn make_image_body(file: std.fs.File, image_height: u16, image_width: u16) !void {
    for (0..image_height) |hi| {
        const hint: u16 = @intCast(hi);
        const h: f32 = @floatFromInt(hint);
        for (0..image_width) |wi| {
            const wint: u16 = @intCast(wi);
            const w: f32 = @floatFromInt(wint);
            const w_denom: f32 = @floatFromInt(image_width - 1);
            const h_denom: f32 = @floatFromInt(image_height - 1);

            const r: f32 = w / w_denom;
            const g: f32 = h / h_denom;
            const b: f32 = 0;

            const ir: u8 = @intFromFloat(255.999 * r);
            const ig: u8 = @intFromFloat(255.999 * g);
            const ib: u8 = @intFromFloat(255.999 * b);

            var buffer: [128]u8 = undefined;
            const formatted_string = try std.fmt.bufPrint(&buffer, "{} {} {}\n", .{ ir, ig, ib });
            try file.writeAll(formatted_string);
        }
    }
}
