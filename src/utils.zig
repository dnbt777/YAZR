const std = @import("std");
const print = std.debug.print;

pub fn dot(a: @Vector(3, f32), b: @Vector(3, f32)) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn cross(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{
        a[1] * b[2] - a[2] * b[1],
        -(a[0] * b[2] - a[2] * b[0]),
        a[0] * b[1] - a[1] * b[0],
    });
}

pub fn splat(x: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), @splat(x));
}

pub fn linear_to_gamma(linear_component: f32) f32 {
    if (linear_component > 0) {
        return std.math.sqrt(linear_component);
    }
    return 0;
}

pub fn reflect(vec: @Vector(3, f32), n: @Vector(3, f32)) @Vector(3, f32) {
    return vec - splat(2 * dot(vec, n)) * n;
}

pub fn refract(uv: @Vector(3, f32), n: @Vector(3, f32), etai_over_etat: f32) @Vector(3, f32) {
    const cos_theta = @min(dot(-uv, n), 1.0);
    const r_out_perp = splat(etai_over_etat) * (uv + splat(cos_theta) * n);
    const r_out_parallel = splat(-std.math.sqrt(@abs(1 - dot(r_out_perp, r_out_perp)))) * n;
    return r_out_perp + r_out_parallel;
}

pub fn vec3(x: f32, y: f32, z: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{ x, y, z });
}

pub fn near_zero(vec: @Vector(3, f32)) bool {
    const s = 1e-8;
    if (@abs(vec[0]) >= s) {
        return false;
    } else if (@abs(vec[1]) >= s) {
        return false;
    } else if (@abs(vec[2]) >= s) {
        return false;
    }
    return true;
}

pub fn random_vec3() @Vector(3, f32) {
    const vec = vec3(random_double(), random_double(), random_double());
    return vec;
}

pub fn random_vec3_range(min: f32, max: f32) @Vector(3, f32) {
    const vec = vec3(random_double_range(min, max), random_double_range(min, max), random_double_range(min, max));
    // print("\n{} {} {}", .{ vec[0], vec[1], vec[2] });
    return vec;
}

pub fn random_in_unit_sphere() @Vector(3, f32) {
    while (true) {
        const p = random_vec3_range(-1, 1);
        if (dot(p, p) < 1) {
            return p;
        }
    }
}

pub fn random_unit_vector() @Vector(3, f32) {
    return unit_vector(random_in_unit_sphere());
}

pub fn random_in_unit_disk() @Vector(3, f32) {
    while (true) {
        const p = vec3(random_double_range(-1, 1), random_double_range(-1, 1), 0);
        if (dot(p, p) < 1) {
            return p;
        }
    }
}

pub fn random_on_hemisphere(normal: @Vector(3, f32)) @Vector(3, f32) {
    const on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

pub fn unit_vector(vec: @Vector(3, f32)) @Vector(3, f32) {
    const x2: f32 = std.math.pow(f32, vec[0], 2);
    const y2: f32 = std.math.pow(f32, vec[1], 2);
    const z2: f32 = std.math.pow(f32, vec[2], 2);
    return vec / splat(std.math.sqrt(x2 + y2 + z2));
}

pub fn degrees_to_radians(degrees: f32) f32 {
    return degrees * std.math.pi / 180.0;
}

pub fn random_double() f32 {
    //millis is not precise enough...
    const timestamp: u64 = @as(u64, @intCast(std.time.nanoTimestamp()));
    var prng = std.Random.DefaultPrng.init(timestamp);
    const rand = prng.random();
    return rand.float(f32);
}

pub fn random_double_range(min: f32, max: f32) f32 {
    return min + (max - min) * random_double();
}
