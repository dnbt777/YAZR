const std = @import("std");
const time = std.time.milliTimestamp;
const nanotime = std.time.nanoTimestamp;
const display = @import("display.zig");
const c = @cImport({
    @cInclude("X11/Xlib.h");
});

// structs probbaly need to match exactly what is in cuda
const Sphere = extern struct {
    center: [3]f32,
    radius: f32,
};

const PhysicsProperties = extern struct {
    velocity: @Vector(3, f32), // vector if the CPU works with it, [3]f32 if its sent to GPU.
    mass: f32,
    acceleration: @Vector(3, f32), // in this case the physics is done CPU sided
    position: @Vector(3, f32),
};

const PhysicsObjects = extern struct {
    geometries: [256]Sphere, // only spheres for now. this is what gets sent to the renderer
    physical_properties: [256]PhysicsProperties,
};

const Ray = struct {
    tmin: f32,
    tmax: f32,
    direction: [3]f32,
    origin: [3]f32,
};

const HitRecord = struct {
    p: [3]f32,
    normal: [3]f32,
    t: f32,
    front_face: bool,
};

// ok so, if we're operating on a matrix, why is M a pointer to a single f32?
// the reason is bc cuda uses the first pointer to find the start of the matrix
// then it uses width and height to move over in memory to the other parts
// hence why we call fillMat(&r[0]..  rather than fillMat(&r..
// we dont pass in the whole R channel, just a pointer to its first value
extern "c" fn initScene(
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
    level_geometry_host: *Sphere, // pass in level_geometry_host[0]. in cuda the signature is Sphere* indicating array
    level_objects_host: *Sphere,
    static_obj_count: u16,
    dynamic_obj_count: u16,
) u16; // returns failure code or w/e
extern "c" fn render_scene(
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
    level_objects_host: *Sphere,
    dynamic_obj_count: u16,
) void;

//extern "c" fn renderKernel(imageR, imageG, imageB, width, height, samples_per_pixel, depth, hittables_flattened, num_hittables)

// extern "C" void shootrays{
//     - move hittables and image to device, keep em there forever ig (between frames)
//     - for each pixel shoot 500 rays or whatever
//     - for each ray
//         - check all hittables and keep track of first intersection:
//     - if it keeps bounding make it stop if bounces=depth or whatever

// shootrays is just render() ig? or.. i mean, it calculates what needs to be rendered but it is not writing to io

// utils.zig
pub fn splat(x: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{ x, x, x });
}

pub fn vec3(x: f32, y: f32, z: f32) @Vector(3, f32) {
    return @as(@Vector(3, f32), .{ x, y, z });
}

pub fn unit_vector(v: @Vector(3, f32)) @Vector(3, f32) {
    return v / splat(@sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
}

pub fn ray_position(r: Ray, t: f32) @Vector(3, f32) {
    return r.origin + splat(t) * r.direction;
}

pub fn cross(a: @Vector(3, f32), b: @Vector(3, f32)) @Vector(3, f32) {
    return vec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

const Hittable = struct {
    // pub fn hit(r : *Ray, ray_tmin: f32, ray_tmax: f32, rec: HitRecord) bool {
    //     return false;
    // }
};

pub fn main() !void {
    // unused bc opengl now
    // var allocator = std.heap.page_allocator;

    const N = 1000; // square for now
    const image_height = N;
    const image_width = N;
    const aspect_ratio: f32 = @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));

    const static_obj_count = 4;
    var level_geometry_host: [256]Sphere = undefined;
    level_geometry_host[0] = Sphere{
        .center = .{ 0.0, 0.0, 0.5 },
        .radius = 0.5,
    };
    level_geometry_host[1] = Sphere{
        .center = .{ 0.0, 0.0, 2.0 },
        .radius = 0.5,
    };
    level_geometry_host[2] = Sphere{
        .center = .{ 2.0, 0.0, 0.0 },
        .radius = 0.5,
    };
    level_geometry_host[3] = Sphere{
        .center = .{ 0.0, -100.5, -1.0 },
        .radius = 100.0,
    };

    // const PhysicsProperties = extern struct {
    //     velocity: [3]f32,
    //     mass: [3]f32,
    //     acceleration: [3]f32,
    // };
    //
    // const PhysicsObjects = extern struct {
    //     geometries: [256]Sphere, // only spheres for now. this is what gets sent to the renderer
    //     physical_properties: [256]PhysicsProperties,
    // }

    const physics_obj_count = 100;
    var physics_objects: PhysicsObjects = undefined;
    for (0..physics_obj_count) |obj_idx| {
        const offset = @as(f32, @floatFromInt(obj_idx));
        const radius = 1.0 + offset;
        const row = @mod(offset, 10.0);
        const col = offset / @as(f32, @floatFromInt(physics_obj_count));
        physics_objects.geometries[obj_idx] = Sphere{
            .center = .{
                0.0 + col * radius * offset / 2,
                40.0,
                0.0 + row * radius * offset / 2,
            },
            .radius = radius,
        };

        physics_objects.physical_properties[obj_idx] = PhysicsProperties{
            .velocity = splat(0.0),
            .mass = 1.0,
            .acceleration = vec3(0, @min(-1, -9.8 + offset), 0),
            .position = physics_objects.geometries[obj_idx].center,
        };
    }

    // init scene
    const start = time();

    // get viewport/camera stuff
    const focal_length: f32 = 1.0;
    const viewport_height = 2.0;
    const viewport_width = viewport_height * aspect_ratio; // just doing it this way for now
    const mouse_sensitivity: f32 = 20;
    var origin: @Vector(3, f32) = .{ 80.0, 80.0, 80.0 };
    // var look_at: @Vector(3, f32) = vec3(0.0, 0.0, 0.0); // look at the origin for now
    var theta_horizontal: f32 = 0.0;
    var theta_vertical: f32 = 0.0;
    var forward = unit_vector(vec3(
        @sin(theta_horizontal), //left/right
        @sin(theta_vertical), //up/dowj
        @cos(theta_horizontal), //back/forth
    )); // OPTIMIZATION: remove unit vector function, its already a unit vector, although idc abt init time tbh
    var right = unit_vector(cross(forward, vec3(0.0, 1.0, 0.0))); // y is world up here
    var up = unit_vector(cross(right, forward)); // thanks ai..?
    var viewport_u = splat(viewport_width) * right; //vec3(viewport_width, 0, 0);
    var viewport_v = splat(-viewport_height) * up; //may not be negative // vec3(0, -viewport_height, 0);
    var pixel_delta_u = viewport_u / splat(@as(f32, @floatFromInt(image_width)));
    var pixel_delta_v = viewport_v / splat(@as(f32, @floatFromInt(image_height)));
    var viewport_upper_left = origin + splat(focal_length) * forward - splat(0.5) * viewport_u - splat(0.5) * viewport_v;
    var pixel00_loc = viewport_upper_left + splat(0.5) * (pixel_delta_u + pixel_delta_v);
    // var forward = unit_vector(origin - look_at);
    // var right = unit_vector(cross(vec3(0.0, 0.0, 1.0), forward));
    // var up = unit_vector(cross(forward, right)); // thanks ai..?
    // var viewport_u = splat(viewport_width) * right; //vec3(viewport_width, 0, 0);
    // var viewport_v = splat(-viewport_height) * up; //may not be negative // vec3(0, -viewport_height, 0);
    // var pixel_delta_u = viewport_u / splat(@as(f32, @floatFromInt(image_width)));
    // var pixel_delta_v = viewport_v / splat(@as(f32, @floatFromInt(image_height)));
    // var viewport_upper_left = origin - vec3(0, 0, focal_length) - splat(0.5) * viewport_u - splat(0.5) * viewport_v;
    // var pixel00_loc = viewport_upper_left + splat(0.5) * (pixel_delta_u + pixel_delta_v);
    const samples = 1;

    // returns an int that must be discarded
    _ = initScene(
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
        &level_geometry_host[0],
        &physics_objects.geometries[0],
        static_obj_count,
        physics_obj_count,
    );

    // i had an llm generate this block
    const xdisplay = c.XOpenDisplay(null);
    defer _ = c.XCloseDisplay(xdisplay); // returns a val so must be discarded into _
    if (xdisplay == null) {
        std.debug.print("Failed to open X display\n", .{});
        return;
    }

    var root: c.Window = 0;
    var child: c.Window = 0;
    var root_x: c_int = 0;
    var root_y: c_int = 0;
    var win_x: c_int = 0;
    var win_y: c_int = 0;
    var mask: c_uint = 0;
    var last_mouse_x: c_int = 0;
    var last_mouse_y: c_int = 0;
    var mouse_dx: f32 = undefined;
    var mouse_dy: f32 = undefined;

    const screen = c.XDefaultScreen(xdisplay);
    const root_window = c.XRootWindow(xdisplay, screen);

    // _ = c.XSelectInput(xdisplay, root_window, c.KeyPressMask | c.KeyReleaseMask);
    const return_events: c_int = 0;
    _ = c.XGrabKeyboard(xdisplay, root_window, return_events, c.GrabModeAsync, c.GrabModeAsync, c.CurrentTime);
    // https://stackoverflow.com/questions/2792954/x11-how-do-i-really-grab-the-mouse-pointer
    _ = c.XGrabPointer(xdisplay, root_window, return_events, c.ButtonPressMask |
        c.ButtonReleaseMask |
        c.PointerMotionMask |
        c.FocusChangeMask |
        c.EnterWindowMask |
        c.LeaveWindowMask, c.GrabModeAsync, c.GrabModeAsync, root_window, c.None, c.CurrentTime);
    if (c.XQueryPointer(xdisplay, root_window, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask) == 0) {
        std.debug.print("Failed to query pointer\n", .{});
        return;
    }

    var event: c.XEvent = undefined;
    std.debug.print("Scene init time: {}ms\n", .{time() - start});

    // shoot rays and send to images
    // make things pulse at multiples of 60bpm.. this makes it so whatever music ppl listen to, the game
    // dances to it
    const start2 = time();
    var quit: bool = false;
    var dx: f32 = 0;
    var dy: f32 = 0;
    var dz: f32 = 0;
    var last_time: i64 = time();
    var current_time: i64 = undefined;
    var dt: f32 = undefined;
    for (0..60000) |_| {
        if (quit) {
            break;
        }
        // time
        current_time = time();
        dt = @as(f32, @floatFromInt(current_time - last_time)) / 1000.0; // time is in ms, dt is in seconds
        last_time = current_time;

        //mouse
        last_mouse_x = root_x;
        last_mouse_y = root_y;
        if (c.XQueryPointer(xdisplay, root_window, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask) == 0) {
            std.debug.print("Failed to query pointer\n", .{});
            return;
        }
        mouse_dy = @as(f32, @floatFromInt(root_y - last_mouse_y)) / @as(f32, @floatFromInt(image_height));
        mouse_dx = @as(f32, @floatFromInt(root_x - last_mouse_x)) / @as(f32, @floatFromInt(image_width));
        //std.debug.print("Mouse Deltas: ({}, {})\n", .{ mouse_dx, mouse_dy });

        // keyboard
        // https://stackoverflow.com/questions/20489449/how-do-i-get-events-in-the-x-window-system-without-pausing-execution
        // https://gist.github.com/javiercantero/7753445
        // dx = 0;
        // dy = 0;
        // dz = 0;
        while (c.XPending(xdisplay) != 0) {
            _ = c.XNextEvent(xdisplay, &event);

            std.debug.print("{}\n", .{event.type});
            if (event.type == c.KeyPress) {
                std.debug.print("KeyPress: {}\n", .{event.xkey.keycode});
                if (event.xkey.keycode == 0x09) { // escape key
                    quit = true;
                    break;
                }
                switch (event.xkey.keycode) {
                    0x28 => dx = -0.1,
                    0x26 => dx = 0.1,
                    0x32 => dy = -0.1,
                    0x41 => dy = 0.1,
                    0x19 => dz = 0.1,
                    0x27 => dz = -0.1,
                    0x09 => quit = true,
                    else => dx = 0,
                }
            } else if (event.type == c.KeyRelease) {
                std.debug.print("KeyRelease: {}\n", .{event.xkey.keycode});
                if (event.xkey.keycode == 0x09) { // escape key
                    quit = true;
                    break;
                }
                switch (event.xkey.keycode) {
                    0x28 => dx = 0,
                    0x26 => dx = 0,
                    0x32 => dy = 0,
                    0x41 => dy = 0,
                    0x19 => dz = 0,
                    0x27 => dz = 0,
                    0x09 => quit = true,
                    else => dx = 0,
                }
            } else if (event.type == c.ButtonPress) {
                std.debug.print("Pew!", .{});
            }
        }

        // update game
        const speed = 5.0;
        // update physics
        for (0..physics_obj_count) |obj_idx| {
            physics_objects.physical_properties[obj_idx].velocity += physics_objects.physical_properties[obj_idx].acceleration * splat(dt);
            physics_objects.physical_properties[obj_idx].position += physics_objects.physical_properties[obj_idx].velocity * splat(dt);
        }

        // update geometries based on physics properties
        // assume all spheres for now.
        for (0..physics_obj_count) |obj_idx| {
            // if obj.type == sphere
            physics_objects.geometries[obj_idx].center = physics_objects.physical_properties[obj_idx].position;
        }

        // render
        theta_horizontal += mouse_sensitivity * mouse_dx;
        theta_vertical -= mouse_sensitivity * mouse_dy;
        forward = unit_vector(vec3(
            @sin(theta_horizontal), //left/right
            @sin(theta_vertical), //up/dowj
            @cos(theta_horizontal), //back/forth
        )); // (OPTIMIZATION: remove unit_vector call, its already a unit vector)
        right = unit_vector(cross(forward, vec3(0.0, 1.0, 0.0))); // y is world up here
        up = unit_vector(cross(right, forward)); // thanks ai..?
        viewport_u = splat(-viewport_width) * right; //vec3(viewport_width, 0, 0);
        viewport_v = splat(viewport_height) * up; //may not be negative // vec3(0, -viewport_height, 0);
        pixel_delta_u = viewport_u / splat(@as(f32, @floatFromInt(image_width)));
        pixel_delta_v = viewport_v / splat(@as(f32, @floatFromInt(image_height)));
        origin += splat(dx * speed) * right + splat(dy * speed) * up + splat(dz * speed) * forward; // player position
        viewport_upper_left = origin + splat(focal_length) * forward - splat(0.5) * viewport_u - splat(0.5) * viewport_v;
        pixel00_loc = viewport_upper_left + splat(0.5) * (pixel_delta_u + pixel_delta_v);

        render_scene(
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
            &physics_objects.geometries[0], // pass all dynamic objects to device every frame. allows them to be updated
            physics_obj_count,
        );
    }
    const duration = time() - start2; // ms
    std.debug.print("Ray shooting time: {}ms\n", .{duration});
    const seconds = @as(f32, @floatFromInt(duration)) / 1000.0;
    std.debug.print("Avg FPS: {} FPS\n", .{1200.0 / seconds});
    // Writes to ppm. alternate in future: write to some screen buffer or something
    const start3 = time();
    // opengl, we aint doin this no more
    // try display.write_image(&image, image_width, image_height);
    std.debug.print("Image writing time: {}ms\n", .{time() - start3});
}
