const std = @import("std");
const time = std.time.milliTimestamp;
const display = @import("display.zig");

// ok so, if we're operating on a matrix, why is M a pointer to a single f32?
// the reason is bc cuda uses the first pointer to find the start of the matrix
// then it uses width and height to move over in memory to the other parts
// hence why we call fillMat(&r[0]..  rather than fillMat(&r..
// we dont pass in the whole R channel, just a pointer to its first value
extern "c" fn fillMat(M: *const f32, width: u16, height: u16, c: f32) void;
extern "c" fn colorImg(R: *const f32, G: *const f32, B: *const f32, width: u16, height: u16) void;
extern "c" fn initImage(R: *const f32, G: *const f32, B: *const f32, width: u16, height: u16) void;
extern "c" fn shootRays(
    R: *const f32,
    G: *const f32,
    B: *const f32,
    width: u16,
    height: u16,
    pixel_delta_u0: f32, // degenerate way to do it but.. whatever
    pixel_delta_u1: f32,
    pixel_delta_u2: f32,
    pixel_delta_v0: f32,
    pixel_delta_v1: f32,
    pixel_delta_v2: f32,
    pixel00_loc0: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc1: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    pixel00_loc2: f32, // not sure if this should be a slice.. do i need to move this to device, or change signature?
    origin0: f32, // not sure if this should be a slice...
    origin1: f32, // not sure if this should be a slice...
    origin2: f32, // not sure if this should be a slice...
    samples: u16,
) void;

//extern "c" fn shootRaysKernel(imageR, imageG, imageB, width, height, samples_per_pixel, depth, hittables_flattened, num_hittables)

// extern "C" void shootrays{
//     - move hittables and image to device, keep em there forever ig (between frames)
//     - for each pixel shoot 500 rays or whatever
//     - for each ray
//         - check all hittables and keep track of first intersection:
//     - if it keeps bounding make it stop if bounces=depth or whatever

// shootrays is just render() ig? or.. i mean, it calculates what needs to be rendered but it is not writing to io

// do i even need this????
const Ray = struct {
    origin: @Vector(3, f32),
    direction: @Vector(3, f32),
};

// utils.zig
pub fn splat(c: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{ c, c, c });
}

pub fn vec3(x: f32, y: f32, z: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{ x, y, z });
}

pub fn ray_position(r: Ray, t: f32) @Vector(3, f32) {
    return r.origin + splat(t) * r.direction;
}

pub fn main() !void {
    var allocator = std.heap.page_allocator;

    const N = 1000; // square for now
    const image_height = N;
    const image_width = N;
    const aspect_ratio: f32 = @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));

    // init image w r g and b channels
    const start = time();
    var image: [3][]f32 = .{
        try allocator.alloc(f32, image_height * image_width),
        try allocator.alloc(f32, image_height * image_width),
        try allocator.alloc(f32, image_height * image_width),
    };
    initImage(&image[0][0], &image[1][0], &image[2][0], image_width, image_height);
    std.debug.print("Allocation time: {}ms\n", .{time() - start});

    // if this works it will output a slightly red/grey image
    // fillMat(&image[0][0], image_width, image_height, 254.0);
    // fillMat(&image[1][0], image_width, image_height, 100.0);
    // fillMat(&image[2][0], image_width, image_height, 100.0);

    // colorImg(&image[0][0], &image[1][0], &image[2][0], image_width, image_height);

    // get viewport/camera stuff
    const focal_length = 1.0;
    const viewport_height = 2.0;
    const viewport_width = viewport_height * aspect_ratio; // just doing it this way for now
    const camera_center: @Vector(3, f32) = .{ 0, 0, 0 };
    const viewport_u = vec3(viewport_width, 0, 0);
    const viewport_v = vec3(0, -viewport_height, 0);
    const pixel_delta_u = viewport_u / splat(@as(f32, @floatFromInt(image_width)));
    const pixel_delta_v = viewport_v / splat(@as(f32, @floatFromInt(image_height)));
    const viewport_upper_left = camera_center - vec3(0, 0, focal_length) - splat(0.5) * viewport_u - splat(0.5) * viewport_v;
    const pixel00_loc = viewport_upper_left + splat(0.5) * (pixel_delta_u + pixel_delta_v);

    // shoot rays and send to images
    const start2 = time();
    const origin = camera_center; // just do this for now
    const samples = 500;
    for (0..60) |_| {
        shootRays(
            &image[0][0],
            &image[1][0],
            &image[2][0],
            image_width,
            image_height,
            pixel_delta_u[0],
            pixel_delta_u[1],
            pixel_delta_u[2],
            pixel_delta_v[0],
            pixel_delta_v[1],
            pixel_delta_v[2],
            pixel00_loc[0],
            pixel00_loc[1],
            pixel00_loc[2],
            origin[0],
            origin[1],
            origin[2],
            samples,
        );
    }
    std.debug.print("Ray shooting time: {}ms\n", .{time() - start2});

    // Writes to ppm. alternate in future: write to some screen buffer or something
    const start3 = time();
    try display.write_image(&image, image_width, image_height);
    std.debug.print("Image writing time: {}ms\n", .{time() - start3});
}
