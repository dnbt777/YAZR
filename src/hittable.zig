const interval = @import("interval.zig");
const utils = @import("utils.zig");
const std = @import("std");
const rt = @import("rt.zig");
const material = @import("material.zig");

pub fn hit_sphere(center: @Vector(3, f32), radius: f32, ray: rt.Ray) f32 {
    const oc: @Vector(3, f32) = center - ray.origin;
    const a = utils.dot(ray.direction, ray.direction);
    const h = utils.dot(ray.direction, oc);
    const c = utils.dot(oc, oc) - radius * radius;
    const discriminant = h * h - a * c;
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - std.math.sqrt(discriminant)) / a;
    }
}

pub const HittableList = struct {
    objects: std.ArrayList(Hittable),
    pub fn clear(self: HittableList) void {
        self = self; // placeholder TODO
    }
    pub fn add(self: *HittableList, object: Hittable) !void {
        try self.objects.append(object);
    }
    pub fn hit(self: HittableList, ray: rt.Ray, ray_t: interval.Interval, hit_record: *HitRecord) bool {
        var temp_record: HitRecord = hit_record.*;
        var hit_anything: bool = false;
        var closest_so_far: f32 = ray_t.max;

        for (self.objects.items) |object| {
            const closest_interval: interval.Interval = interval.Interval{ .min = ray_t.min, .max = closest_so_far };
            if (object.hit(ray, closest_interval, &temp_record)) {
                hit_anything = true;
                closest_so_far = temp_record.t;
                hit_record.* = temp_record;
            }
        }

        return hit_anything;
    }
};

pub const Sphere = struct {
    //const mat: *const material.Material = &material.metal;
    mat: *const material.Material,
    center: @Vector(3, f32),
    radius: f32,
    pub fn hit(self: Sphere, ray: rt.Ray, ray_t: interval.Interval, hit_record: *HitRecord) bool {
        const oc: @Vector(3, f32) = self.center - ray.origin;
        const a = utils.dot(ray.direction, ray.direction);
        const h = utils.dot(ray.direction, oc);
        const c = utils.dot(oc, oc) - self.radius * self.radius;
        const discriminant = h * h - a * c;
        if (discriminant < 0) {
            // return -1.0;
            return false;
        }

        const sqrtd: f32 = std.math.sqrt(discriminant);
        var root: f32 = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        hit_record.t = root;
        hit_record.p = ray.at(hit_record.t);
        const outward_normal: @Vector(3, f32) = (hit_record.p - self.center) / utils.splat(self.radius);
        hit_record.set_face_normal(ray, outward_normal);
        hit_record.mat = self.mat;
        return true;
    }
};

pub const HitRecord = struct {
    p: @Vector(3, f32),
    normal: @Vector(3, f32),
    t: f32,
    front_face: bool,
    mat: *const material.Material,
    pub fn set_face_normal(self: *HitRecord, ray: rt.Ray, outward_normal: @Vector(3, f32)) void {
        self.front_face = utils.dot(ray.direction, outward_normal) < 0;
        self.normal = if (self.front_face) outward_normal else -outward_normal;
    }
};

pub const Hittable_Tags = enum { sphere, hittable_list };

pub const Hittable = union(Hittable_Tags) { //union(enum) also works and is equivalent to the above
    sphere: Sphere,
    hittable_list: HittableList,
    pub fn hit(self: Hittable, ray: rt.Ray, ray_t: interval.Interval, hit_record: *HitRecord) bool {
        const hitsomething: bool = switch (self) {
            .sphere => |object| object.hit(ray, ray_t, hit_record),
            .hittable_list => |object| object.hit(ray, ray_t, hit_record),
        };
        return hitsomething; // empty list or obj not implemented
    }
};
