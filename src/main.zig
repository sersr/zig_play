const std = @import("std");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const builtin = @import("builtin");
const v = @import("vulkan.zig");

// const glfw = @cImport({
//     @cDefine("GLFW_INCLUDE_VULKAN", "");
//     @cInclude("GLFW/glfw3.h");
// });

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
pub fn main() !void {
    // std.debug.print("helo", .{});
    // try read_elf();
    try glfwMain();
}

fn read_elf() !void {
    const elf_file = try std.fs.cwd().openFile("go_test", .{});
    const size = (try elf_file.metadata()).size();

    const buffer = try elf_file.readToEndAlloc(std.heap.page_allocator, size);
    defer std.heap.page_allocator.free(buffer);
    defer std.debug.print("free", .{});

    var source = std.io.StreamSource{ .buffer = std.io.fixedBufferStream(buffer) };

    const header = try std.elf.Header.read(&source);

    std.log.info("header mag {?}", .{header});

    var pro = header.program_header_iterator(&source);
    var current = try pro.next();
    while (current != null) {
        std.log.info("current: {?}", .{current});
        current = try pro.next();
    }

    std.debug.print("helo{}", .{1});
    var sec = header.section_header_iterator(&source);
    var sec_current = try sec.next();

    while (sec_current != null) {
        std.log.info("section: {?}", .{sec_current});
        sec_current = try sec.next();
    }
}

fn glfwMain() !void {
    std.debug.print("glfw init\n", .{});

    var vp = try v.init(800, 600);
    // defer vp.deinit();
    try vp.run();
}
