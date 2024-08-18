pub const Ray = struct {
    origin: @Vector(3, f32),
    direction: @Vector(3, f32),
    pub fn at(self: *const Ray, t: f32) @Vector(3, f32) {
        return self.origin + @as(@Vector(3, f32), @splat(t)) * self.direction;
    }
};
