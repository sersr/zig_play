const std = @import("std");
const builtin = @import("builtin");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exeOption = b.option([]const u8, "exe", "");
    const sub = b.option(bool, "sub", "subsystem");

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

    const zglfw = b.dependency("zglfw", .{});
    exe.root_module.addImport("zglfw", zglfw.module("glfw"));

    if (sub) |s| {
        if (s) {
            exe.subsystem = .Windows;
        }
    }

    // Get the (lazy) path to vk.xml:
    const vulkan_header = b.dependency("vulkan_headers", .{});
    const registry = vulkan_header.path("registry/vk.xml");

    const vulkan_zig = b.dependency("vulkan", .{
        .registry = registry,
    }).module("vulkan-zig");

    exe.root_module.addImport("vulkan", vulkan_zig);

    switch (builtin.target.os.tag) {
        .windows => {
            const glfw_lib = b.dependency("glfw_lib_win", .{});
            const glfw_lib_root = glfw_lib.path("lib-static-ucrt");
            const glfw_lib_path = glfw_lib_root.path(b, "glfw3.dll");

            const file = b.addInstallFileWithDir(glfw_lib_path, .bin, "glfw3.dll");
            b.getInstallStep().dependOn(&file.step);
            exe.addLibraryPath(glfw_lib_root);
        },
        .macos => {
            const glfw_lib = b.dependency("glfw_lib_mac", .{});
            const lib_path = switch (builtin.cpu.arch) {
                .aarch64 => "lib-arm64",
                .x86_64 => "lib-x86_64",
                else => "lib-universal",
            };

            const glfw_lib_root = glfw_lib.path(lib_path);
            const glfw_lib_path = glfw_lib_root.path(b, "libglfw.3.dylib");

            const file = b.addInstallFileWithDir(glfw_lib_path, .bin, "libglfw.3.dylib");
            b.getInstallStep().dependOn(&file.step);
            exe.addLibraryPath(glfw_lib_root);
        },
        else => {},
    }

    exe.linkSystemLibrary("glfw3");
    exe.linkLibC();

    const zmath = b.dependency("zmath", .{}).module("root");
    exe.root_module.addImport("zmath", zmath);

    const zstbi = b.dependency("zstbi", .{}).module("root");
    exe.root_module.addImport("stbi", zstbi);
    const stb_c = b.dependency("stb", .{}).path("");
    exe.root_module.addIncludePath(stb_c);

    const tinyobjloader = b.dependency("tinyobjloader", .{}).path("");

    exe.root_module.addIncludePath(tinyobjloader);
    exe.addCSourceFile(.{
        .file = b.path("src/tiny_obj_loader.c"),
        .language = .c,
    });

    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);

    run.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run.addArgs(args);
    }
    const step = b.step("run", "");
    step.dependOn(&run.step);
}
