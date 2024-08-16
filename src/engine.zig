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
    lookfrom: @Vector(3, f32),
    lookat: @Vector(3, f32),
    vup: @Vector(3, f32),

    v: @Vector(3, f32),
    u: @Vector(3, f32),
    w: @Vector(3, f32),

    defocus_angle: f32,
    focus_dist: f32,

    defocus_disk_u: @Vector(3, f32),
    defocus_disk_v: @Vector(3, f32),

    pub fn init(self: *Camera) void {
        self.camera_center = self.lookfrom;
        self.image_height = @intFromFloat(@as(f32, @floatFromInt(self.image_width)) / self.aspect_ratio);
        const theta = utils.degrees_to_radians(self.vfov);
        const h = std.math.tan(theta / 2.0);
        self.viewport_height = 2.0 * h * self.focus_dist;
        self.viewport_aspect_ratio = @as(f32, @floatFromInt(self.image_width)) / @as(f32, @floatFromInt(self.image_height));
        self.viewport_width = self.viewport_height * self.viewport_aspect_ratio;
        self.w = utils.unit_vector(self.lookfrom - self.lookat);
        self.u = utils.unit_vector(utils.cross(self.vup, self.w));
        self.v = utils.cross(self.w, self.u);
        self.viewport_u = utils.splat(self.viewport_width) * self.u;
        self.viewport_v = utils.splat(self.viewport_height) * -self.v;
        self.pixel_delta_u = self.viewport_u / utils.splat(@as(f32, @floatFromInt(self.image_width)));
        self.pixel_delta_v = self.viewport_v / utils.splat(@as(f32, @floatFromInt(self.image_height)));
        self.viewport_upper_left = self.camera_center - utils.splat(self.focus_dist) * self.w - self.viewport_u / utils.splat(2.0) - self.viewport_v / utils.splat(2.0);
        self.pixel00_loc = self.viewport_upper_left + utils.splat(0.5) * (self.pixel_delta_u + self.pixel_delta_v);
        self.pixel_samples_scale = 1.0 / @as(f32, @floatFromInt(self.samples_per_pixel));

        const defocus_radius = self.focus_dist * std.math.tan(utils.degrees_to_radians(self.defocus_angle / 2.0));
        self.defocus_disk_u = self.u * utils.splat(defocus_radius);
        self.defocus_disk_v = self.v * utils.splat(defocus_radius);
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

        const ray_origin = if (self.defocus_angle <= 0) self.camera_center else self.defocus_disk_sample();
        const ray_direction = pixel_sample - ray_origin;
        return rt.Ray{ .origin = ray_origin, .direction = ray_direction };
    }

    pub fn defocus_disk_sample(self: Camera) @Vector(3, f32) {
        const p = utils.random_in_unit_disk();
        return self.camera_center + utils.splat(p[0]) * self.defocus_disk_u + utils.splat(p[1]) * self.defocus_disk_v;
    }
};

pub fn default_camera() Camera {
    var camera: Camera = undefined;

    camera.vfov = 20;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 400;

    camera.samples_per_pixel = 50; // can be huge source of inefficiency.. 1 = 1 pass per frame
    camera.max_depth = 7; // can also be huge source of inefficiency.. not sure what the avg is tho

    camera.lookfrom = utils.vec3(-2, 2, 1);
    camera.lookat = utils.vec3(0, 0, -1);
    camera.vup = utils.vec3(0, 1, 0);

    camera.focal_length = 1.0;
    camera.defocus_angle = 10.0; //0
    camera.focus_dist = 3.4; //10

    _ = &camera.init();
    return camera;
}

pub fn final_render_camera() Camera {
    var camera: Camera = undefined;

    camera.vfov = 20;
    camera.aspect_ratio = 16.0 / 9.0;
    camera.image_width = 200; //1200

    camera.samples_per_pixel = 10; // 500
    camera.max_depth = 7; // 50

    camera.lookfrom = utils.vec3(13, 2, 3);
    camera.lookat = utils.vec3(0, 0, 0);
    camera.vup = utils.vec3(0, 1, 0);

    camera.focal_length = 1.0;
    camera.defocus_angle = 0.6;
    camera.focus_dist = 10.0;

    _ = &camera.init();
    return camera;
}

pub fn sample_square() @Vector(3, f32) {
    return utils.vec3(utils.random_double() - 0.5, utils.random_double() - 0.5, 0);
}
