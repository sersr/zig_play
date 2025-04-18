const std = @import("std");
const builtin = @import("builtin");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    std.debug.print("build start ...\n", .{});
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const exeOption = b.option([]const u8, "exe", "");

    if (exeOption) |e| {
        const arr: []const u8 = "array";
        if (std.mem.eql(u8, e, arr)) {
            // ------ src/array.zig --------
            const arrayExe = b.addExecutable(.{
                .name = "array",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/array.zig"),
                    .target = target,
                    .optimize = optimize,
                }),
            });

            b.installArtifact(arrayExe);
            const arrayRun = b.addRunArtifact(arrayExe);
            arrayRun.step.dependOn(b.getInstallStep());
            if (b.args) |args| {
                arrayRun.addArgs(args);
            }
            const run_step = b.step("run", "");
            run_step.dependOn(&arrayRun.step);
            return;
            // ------ src/array.zig --------
        }
    }

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const httpz = b.dependency("httpz", .{
        .target = target,
        .optimize = optimize,
    });

    const zglfw = b.dependency("zglfw", .{});
    exe.root_module.addImport("zglfw", zglfw.module("glfw"));

    // if (target.result.os.tag != .emscripten) {
    //     exe.linkLibrary(zglfw.artifact("glfw"));
    // }

    exe.root_module.addImport("httpz", httpz.module("httpz"));

    // exe.subsystem = .Windows;

    // Get the (lazy) path to vk.xml:
    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    // Get generator executable reference
    // const vk_gen = b.dependency("vulkan", .{}).artifact("vulkan-zig-generator");
    // // Set up a run step to generate the bindings
    // const vk_generate_cmd = b.addRunArtifact(vk_gen);
    // // Pass the registry to the generator
    // vk_generate_cmd.addFileArg(registry);
    // // Create a module from the generator's output...
    // const vulkan_zig = b.addModule("vulkan", .{
    //     .root_source_file = vk_generate_cmd.addOutputFileArg("vk.zig"),
    // });
    const vulkan_zig = b.dependency("vulkan", .{
        .registry = registry,
    }).module("vulkan-zig");
    // ... and pass it as a module to your executable's build command
    exe.root_module.addImport("vulkan", vulkan_zig);

    const sdk = std.process.getEnvVarOwned(std.heap.page_allocator, "VULKAN_SDK") catch return;
    defer std.heap.page_allocator.free(sdk);
    const vulkan_sdk_path = std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}include", .{ sdk, std.fs.path.sep_str }) catch return;
    defer std.heap.page_allocator.free(vulkan_sdk_path);
    const vulkan_sdk_lib_path = std.fmt.allocPrint(std.heap.page_allocator, "{s}{s}Lib", .{ sdk, std.fs.path.sep_str }) catch return;
    defer std.heap.page_allocator.free(vulkan_sdk_lib_path);
    std.log.info("vukan sdk path: {s}", .{vulkan_sdk_path});

    exe.addLibraryPath(.{ .cwd_relative = "C:\\Users\\aote\\scoop\\apps\\glfw\\current\\lib-static-ucrt" });
    exe.addLibraryPath(.{ .cwd_relative = vulkan_sdk_lib_path });
    exe.addIncludePath(.{ .cwd_relative = vulkan_sdk_path });
    exe.addIncludePath(.{ .cwd_relative = "C:\\Users\\aote\\scoop\\apps\\glfw\\current\\include" });
    exe.linkSystemLibrary("glfw3");
    // vulkan
    exe.linkSystemLibrary("vulkan-1");
    exe.linkLibC();

    const zlm = b.dependency("zlm", .{}).module("zlm");
    exe.root_module.addImport("zlm", zlm);

    b.installArtifact(exe);
    const httpzRun = b.addRunArtifact(exe);
    httpzRun.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        httpzRun.addArgs(args);
    }
    const httpz_run_step = b.step("run", "");
    httpz_run_step.dependOn(&httpzRun.step);
}
