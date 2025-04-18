const std = @import("std");

pub fn main() void {
    const s = [_]u8{ 1, 2, 3 };
    std.debug.print("s.length {d}\n", .{s.len});
    std.debug.print("s.length {d}\n", .{s[s.len - 1]});
}
