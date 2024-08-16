const std = @import("std");
const rt = @import("rt.zig");
const utils = @import("utils.zig");
const hittable = @import("hittable.zig");

pub const Material = struct {
    albedo: @Vector(3, f32),
    fuzz: f32,
    refraction_index: f32,
    mat_id: u8,
    pub fn scatter(self: Material, ray_in: rt.Ray, hit_record: hittable.HitRecord, attenuation: *@Vector(3, f32), scattered: *rt.Ray) bool {
        const output = switch (self.mat_id) {
            0 => no_scattering(self, ray_in, hit_record, attenuation, scattered),
            1 => lambertian_scattering(self, ray_in, hit_record, attenuation, scattered),
            2 => metal_scattering(self, ray_in, hit_record, attenuation, scattered),
            3 => dialectric_scattering(self, ray_in, hit_record, attenuation, scattered),
            else => no_scattering(self, ray_in, hit_record, attenuation, scattered),
        };
        return output;
    }
    pub fn reflectance(self: Material, cosine: f32, refraction_index: f32) f32 {
        switch (self.mat_id) {
            3 => {
                // Schlick d n
                var r0: f32 = (1.0 - refraction_index) / (1.0 + refraction_index);
                r0 = r0 * r0;
                return r0 + (1.0 - r0) * (std.math.pow(f32, 1.0 - cosine, 5));
            },
            else => return 0,
        }
    }
};

fn no_scattering(self: Material, ray_in: rt.Ray, hit_record: hittable.HitRecord, attenuation: *const @Vector(3, f32), scattered: *const rt.Ray) bool {
    _ = self;
    _ = ray_in;
    _ = hit_record;
    _ = attenuation;
    _ = scattered;
    return false;
}

fn lambertian_scattering(self: Material, ray_in: rt.Ray, hit_record: hittable.HitRecord, attenuation: *@Vector(3, f32), scattered: *rt.Ray) bool {
    var scatter_direction = hit_record.normal + utils.random_unit_vector();

    if (utils.near_zero(scatter_direction)) {
        scatter_direction = hit_record.normal;
    }

    _ = ray_in; // remove
    const scattered_ray = rt.Ray{ .origin = hit_record.p, .direction = scatter_direction };
    scattered.* = scattered_ray;
    attenuation.* = self.albedo;
    return true;
}

pub const lambertian = Material{
    .albedo = utils.vec3(1, 1, 1),
    .mat_id = 1, // this is the material type. 1 is lambertian and 2 is metal.
    .fuzz = undefined,
};

// self being a non-reference may cause memory problems or slowdowns...
// why copy the material every time its passed in???? uh.
fn metal_scattering(self: Material, ray_in: rt.Ray, hit_record: hittable.HitRecord, attenuation: *@Vector(3, f32), scattered: *rt.Ray) bool {
    var reflected = utils.reflect(ray_in.direction, hit_record.normal);
    reflected = utils.unit_vector(reflected) + (utils.splat(self.fuzz) * utils.random_unit_vector());
    const scatter_direction = rt.Ray{ .origin = hit_record.p, .direction = reflected };
    scattered.* = scatter_direction;
    attenuation.* = self.albedo;
    return (utils.dot(scattered.direction, hit_record.normal) > 0);
}

pub const metal = Material{ .albedo = utils.vec3(0.8, 0.8, 0.8), .mat_id = 2, .fuzz = 0.3, .refraction_index = undefined };

fn dialectric_scattering(self: Material, ray_in: rt.Ray, hit_record: hittable.HitRecord, attenuation: *@Vector(3, f32), scattered: *rt.Ray) bool {
    attenuation.* = utils.splat(1);
    const ri: f32 = if (hit_record.front_face) (1.0 / self.refraction_index) else self.refraction_index;

    const unit_direction: @Vector(3, f32) = utils.unit_vector(ray_in.direction);
    const cos_theta: f32 = @min(utils.dot(-unit_direction, hit_record.normal), 1.0);
    const sin_theta: f32 = std.math.sqrt(1 - cos_theta * cos_theta);

    const cannot_refract: bool = (ri * sin_theta) > 1.0;
    var direction: @Vector(3, f32) = undefined;

    if (cannot_refract or (self.reflectance(cos_theta, ri) > utils.random_double())) {
        direction = utils.reflect(unit_direction, hit_record.normal);
    } else {
        direction = utils.refract(unit_direction, hit_record.normal, ri);
    }

    scattered.* = rt.Ray{ .origin = hit_record.p, .direction = direction };

    return true;
}

pub const dialectric = Material{ .refraction_index = 0.1, .albedo = undefined, .fuzz = undefined, .mat_id = 3 };
