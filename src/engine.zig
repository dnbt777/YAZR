const rt = @import("rt.zig");
const hittable = @import("hittable.zig");
const utils = @import("utils.zig");
const interval = @import("interval.zig");
const imgwriter = @import("imgwriter.zig");
const material = @import("material.zig");
const std = @import("std");
const fs = std.fs;
const print = std.debug.print;

pub const Camera = struct {
    aspect_ratio: f32,

    image_width: u16,
    image_height: u16,

    viewport_height: f32,
    viewport_aspect_ratio: f32,
    viewport_width: f32,

    focal_length: f32,
    camera_center: @Vector(3, f32),

    viewport_u: @Vector(3, f32),
    viewport_v: @Vector(3, f32),

    pixel_delta_u: @Vector(3, f32),
    pixel_delta_v: @Vector(3, f32),

    viewport_upper_left: @Vector(3, f32),
    pixel00_loc: @Vector(3, f32),

    samples_per_pixel: u32,
    pixel_samples_scale: f32,

    max_depth: u32, //bold

    vfov: f32,

    pub const vfov = 90;
    pub var aspect_ratio = 16.0 / 9.0;
    pub var max_depth = 5;
    pub var camera_center = .{ 0, 0, 0 };
    pub var image_width = 400;
    pub var focal_length = 1.0;
    pub var samples_per_pixel = 50; // can be huge source of inefficiency... 1/samples render speed per frame

    pub var image_height: u16 = @intFromFloat(@as(f32, @floatFromInt(image_width)) / aspect_ratio);
    const theta = utils.degrees_to_radians(vfov);
    const h = std.math.tan(theta / 2);
    pub var viewport_height = 2.0 * h * focal_length;
    pub var viewport_aspect_ratio = @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));
    pub var viewport_width = viewport_height * viewport_aspect_ratio;
    pub var viewport_u = .{ viewport_width, 0, 0 };
    pub var viewport_v = .{ 0, -viewport_height, 0 };
    pub var pixel_delta_u = viewport_u / utils.splat(@as(f32, @floatFromInt(image_width)));
    pub var pixel_delta_v = viewport_v / utils.splat(@as(f32, @floatFromInt(image_height)));
    pub var viewport_upper_left = camera_center - @Vector(3, f32){ 0, 0, focal_length } - viewport_u / utils.splat(2.0) - viewport_v / utils.splat(2.0);
    pub var pixel00_loc = viewport_upper_left + utils.splat(0.5) * (pixel_delta_u + pixel_delta_v);
    pub var pixel_samples_scale = 1.0 / @as(f32, @floatFromInt(samples_per_pixel));

    // delete in a sec
    pub fn initialize(self: *Camera) void {
        self.* = self.*;
    }

    pub fn ray_color(ray: rt.Ray, depth: u32, world: hittable.Hittable) @Vector(3, f32) {
        if (depth <= 0) {
            return utils.splat(0);
        }

        var hit_record: hittable.HitRecord = hittable.HitRecord{ .p = undefined, .normal = undefined, .t = undefined, .front_face = true, .mat = undefined }; // initializing .mat as undefined makes it break lol

        if (world.hit(ray, interval.Interval{ .min = 0.001, .max = std.math.inf(f32) }, &hit_record)) {
            var scattered: rt.Ray = undefined;
            var attenuation: @Vector(3, f32) = undefined;

            if (hit_record.mat.scatter(ray, hit_record, &attenuation, &scattered)) {
                return attenuation * ray_color(scattered, depth - 1, world);
            }

            return utils.splat(0);
        }

        const unit_direction: @Vector(3, f32) = utils.unit_vector(ray.direction);
        const a = 0.9 * (unit_direction[1] + 1); // later, do hit_record.material.reflectance... maybe?
        return utils.splat(1 - a) * utils.vec3(1, 1, 1) + utils.splat(a) * utils.vec3(0.5, 0.7, 1.0);
    }

    pub fn render(self: Camera, world: hittable.Hittable) !void {
        var file = try fs.cwd().createFile(
            "test.ppm",
            .{ .read = true },
        );
        defer file.close();

        //aight now we render

        try imgwriter.write_image_header(file, self.image_width, self.image_height);

        for (0..self.image_height) |j_usize| {
            const j_int: u16 = @intCast(j_usize);
            const j: f32 = @floatFromInt(j_int);

            for (0..self.image_width) |i_usize| {
                const i_int: u32 = @intCast(i_usize);
                const i: f32 = @floatFromInt(i_int);

                var pixel_color = utils.vec3(0, 0, 0);
                for (0..self.samples_per_pixel) |_| {
                    const r = self.get_ray(i, j);
                    pixel_color += Camera.ray_color(r, self.max_depth, world);
                }

                pixel_color *= utils.splat(self.pixel_samples_scale);

                try imgwriter.write_color(file, pixel_color);
            }
        }
    }

    pub fn get_ray(self: Camera, i: f32, j: f32) rt.Ray {
        const offset = sample_square();
        const pixel_sample = self.pixel00_loc + utils.splat(i + offset[0]) * self.pixel_delta_u + utils.splat(j + offset[1]) * self.pixel_delta_v;

        const ray_origin = self.camera_center;
        const ray_direction = pixel_sample - ray_origin;
        return rt.Ray{ .origin = ray_origin, .direction = ray_direction };
    }
};

pub fn sample_square() @Vector(3, f32) {
    return utils.vec3(utils.random_double() - 0.5, utils.random_double() - 0.5, 0);
}
