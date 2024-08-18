const std = @import("std");
const inf = std.math.inf(f32);

pub const Interval = struct {
    min: f32,
    max: f32,
    pub fn size(self: Interval) f32 {
        return self.max - self.min;
    }
    pub fn contains(self: Interval, x: f32) bool {
        return (self.min <= x) and (x <= self.max);
    }
    pub fn surrounds(self: Interval, x: f32) bool {
        return (self.min < x) and (x < self.max);
    }
    pub fn clamp(self: Interval, x: f32) f32 {
        if (x < self.min) {
            return self.min;
        }
        if (x > self.max) {
            return self.max;
        }
        return x;
    }
};

pub const empty: Interval = Interval{ .min = inf, .max = -inf };
pub const universe: Interval = Interval{ .min = -inf, .max = inf };
pub const o_to_inf: Interval = Interval{ .min = 0, .max = inf };
