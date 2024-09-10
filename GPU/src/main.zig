const std = @import("std");
const time = std.time.milliTimestamp;
const nanotime = std.time.nanoTimestamp;
const utils = @import("utils.zig");
const phys = @import("physics.zig");
const render = @import("render.zig");

const c = @cImport({
    @cInclude("X11/Xlib.h");
});

pub fn main() !void {
    const N = 1000; // square for now
    const image_height = N;
    const image_width = N;
    const aspect_ratio: f32 = @as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height));

    // initialize materials
    const material_count = 4;
    var materials_host: [256]phys.Material = undefined;
    materials_host[0] = phys.norm_shaded;
    materials_host[1] = phys.red_albedo;
    materials_host[2] = phys.green_albedo;
    materials_host[3] = phys.blue_albedo;

    // static objects
    const static_obj_count = 4;
    var level_geometry_host: [256]phys.Sphere = undefined;
    level_geometry_host[0] = phys.Sphere{
        .center = .{ 0.0, 0.0, 0.5 },
        .radius = 0.5,
        .material_id = 1,
    };
    level_geometry_host[1] = phys.Sphere{
        .center = .{ 0.0, 0.0, 2.0 },
        .radius = 0.5,
        .material_id = 2,
    };
    level_geometry_host[2] = phys.Sphere{
        .center = .{ 2.0, 0.0, 0.0 },
        .radius = 0.5,
        .material_id = 3,
    };
    level_geometry_host[3] = phys.Sphere{
        .center = .{ 0.0, -100.5, -1.0 },
        .radius = 100.0,
        .material_id = 0, // define these materials. on the CPU. then in init(), send them to a material list. (static, stays on device). then index this material list.
    };

    var physics_obj_count: u16 = 100;
    var physics_objects: phys.PhysicsObjects = undefined;
    for (0..physics_obj_count) |obj_idx| {
        const offset = @as(f32, @floatFromInt(obj_idx));
        const radius = 1.0 + offset;
        const row = @mod(offset, 10.0);
        const col = offset / @as(f32, @floatFromInt(physics_obj_count));
        physics_objects.geometries[obj_idx] = phys.Sphere{
            .center = .{
                0.0 + col * radius * offset / 2,
                40.0,
                0.0 + row * radius * offset / 2,
            },
            .radius = radius,
            .material_id = 0,
        };

        physics_objects.physical_properties[obj_idx] = phys.PhysicsProperties{
            .velocity = utils.splat(0.0),
            .mass = radius * radius * radius, // roughly proportional
            .acceleration = utils.vec3(0, 0.0 * @min(-1, -9.8 * radius), 0),
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
    // var look_at: @Vector(3, f32) = utils.vec3(0.0, 0.0, 0.0); // look at the origin for now
    var theta_horizontal: f32 = 0.0;
    var theta_vertical: f32 = 0.0;
    var forward = utils.unit_vector(utils.vec3(
        @sin(theta_horizontal), //left/right
        @sin(theta_vertical), //up/dowj
        @cos(theta_horizontal), //back/forth
    )); // OPTIMIZATION: remove unit vector function, its already a unit vector, although idc abt init time tbh
    var right = utils.unit_vector(utils.cross(forward, utils.vec3(0.0, 1.0, 0.0))); // y is world up here
    var up = utils.unit_vector(utils.cross(right, forward)); // thanks ai..?
    var viewport_u = utils.splat(viewport_width) * right; //utils.vec3(viewport_width, 0, 0);
    var viewport_v = utils.splat(-viewport_height) * up; //may not be negative // utils.vec3(0, -viewport_height, 0);
    var pixel_delta_u = viewport_u / utils.splat(@as(f32, @floatFromInt(image_width)));
    var pixel_delta_v = viewport_v / utils.splat(@as(f32, @floatFromInt(image_height)));
    var viewport_upper_left = origin + utils.splat(focal_length) * forward - utils.splat(0.5) * viewport_u - utils.splat(0.5) * viewport_v;
    var pixel00_loc = viewport_upper_left + utils.splat(0.5) * (pixel_delta_u + pixel_delta_v);
    const samples = 1;

    // returns an int that must be discarded
    _ = render.initScene(
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
        &materials_host[0],
        material_count,
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
    const return_events: c_int = 0; // aka 'false'

    const screen = c.XDefaultScreen(xdisplay);
    const root_window = c.XRootWindow(xdisplay, screen);

    // take control of keyboard and mouse
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
    for (0..60000) |frame| {
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

        // keyboard
        // https://stackoverflow.com/questions/20489449/how-do-i-get-events-in-the-x-window-system-without-pausing-execution
        // https://gist.github.com/javiercantero/7753445
        while (c.XPending(xdisplay) != 0) {
            _ = c.XNextEvent(xdisplay, &event);

            // std.debug.print("{}\n", .{event.type});
            if (event.type == c.KeyPress) {
                //std.debug.print("KeyPress: {}\n", .{event.xkey.keycode});
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
                // std.debug.print("KeyRelease: {}\n", .{event.xkey.keycode});
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
            } else if (event.type == c.ButtonPress) { // mouse click
                std.debug.print("Pew!", .{});
                physics_obj_count += 1;

                physics_objects.geometries[physics_obj_count - 1] = phys.Sphere{
                    .center = origin, // start it at the camera's origin
                    .radius = 0.5,
                    .material_id = @mod(@as(u16, @intCast(frame)), material_count),
                };

                physics_objects.physical_properties[physics_obj_count - 1] = phys.PhysicsProperties{
                    .velocity = forward * utils.splat(50.0), // shoot the ball forward
                    .mass = 1.0,
                    .acceleration = utils.vec3(0, -9.8, 0),
                    .position = physics_objects.geometries[physics_obj_count - 1].center,
                };
            }
        }

        // update game
        const speed = 5.0;
        // check for collisions and update
        // yes. i know this i o(n^2). i dont care, this is v1
        for (0..physics_obj_count) |obj_idx_1| {
            for (0..obj_idx_1) |obj_idx_2| {
                if (obj_idx_2 == obj_idx_1) {
                    continue;
                }
                // check for collision
                const center_distance = (physics_objects.geometries[obj_idx_1].radius + physics_objects.geometries[obj_idx_2].radius);
                if (utils.vec3_distance(physics_objects.physical_properties[obj_idx_1].position, physics_objects.physical_properties[obj_idx_2].position) <= center_distance * center_distance) {
                    const direction = utils.unit_vector(utils.vec3(
                        physics_objects.physical_properties[obj_idx_2].position[0],
                        physics_objects.physical_properties[obj_idx_2].position[1],
                        physics_objects.physical_properties[obj_idx_2].position[2],
                    ) - utils.vec3(
                        physics_objects.physical_properties[obj_idx_1].position[0],
                        physics_objects.physical_properties[obj_idx_1].position[1],
                        physics_objects.physical_properties[obj_idx_1].position[2],
                    ));
                    const vel1 = physics_objects.physical_properties[obj_idx_1].velocity;
                    const vel2 = physics_objects.physical_properties[obj_idx_2].velocity;
                    const mass1 = physics_objects.physical_properties[obj_idx_1].mass;
                    const mass2 = physics_objects.physical_properties[obj_idx_2].mass;
                    const velocity_change_1 = utils.splat(utils.dot(vel2, direction) * mass2 / mass1) * vel2;
                    const velocity_change_2 = utils.splat(utils.dot(vel1, direction) * mass1 / mass2) * vel1;
                    std.debug.print("v1 {}\n", .{velocity_change_1});
                    std.debug.print("v2 {}\n", .{velocity_change_2});
                    physics_objects.physical_properties[obj_idx_1].velocity += velocity_change_1;
                    physics_objects.physical_properties[obj_idx_2].velocity += velocity_change_2;
                }
            }
        }

        // standad update of physics (gravity, maybe buoyancy in the future
        for (0..physics_obj_count) |obj_idx| {
            physics_objects.physical_properties[obj_idx].velocity += physics_objects.physical_properties[obj_idx].acceleration * utils.splat(dt);
            physics_objects.physical_properties[obj_idx].position += physics_objects.physical_properties[obj_idx].velocity * utils.splat(dt);
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
        theta_vertical = @min(3.14 / 4.0, theta_vertical);
        theta_vertical = @max(-3.14 / 2.0, theta_vertical);
        forward = utils.unit_vector(utils.vec3(
            @sin(1 - theta_vertical) * @sin(theta_horizontal), // back/forth
            @cos(1 - theta_vertical), // up/down
            @sin(1 - theta_vertical) * @cos(theta_horizontal), //left/right
        )); // (OPTIMIZATION: remove unit_vector call, its already a unit vector)
        right = utils.unit_vector(utils.cross(forward, utils.vec3(0.0, 1.0, 0.0))); // y is world up here
        up = utils.unit_vector(utils.cross(right, forward)); // thanks ai..?
        viewport_u = utils.splat(-viewport_width) * right; //utils.vec3(viewport_width, 0, 0);
        viewport_v = utils.splat(viewport_height) * up; //may not be negative // utils.vec3(0, -viewport_height, 0);
        pixel_delta_u = viewport_u / utils.splat(@as(f32, @floatFromInt(image_width)));
        pixel_delta_v = viewport_v / utils.splat(@as(f32, @floatFromInt(image_height)));
        origin += utils.splat(dx * speed) * right + utils.splat(dy * speed) * up + utils.splat(dz * speed) * forward; // player position
        viewport_upper_left = origin + utils.splat(focal_length) * forward - utils.splat(0.5) * viewport_u - utils.splat(0.5) * viewport_v;
        pixel00_loc = viewport_upper_left + utils.splat(0.5) * (pixel_delta_u + pixel_delta_v);

        render.render_scene(
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
