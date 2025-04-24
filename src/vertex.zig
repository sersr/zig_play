const std = @import("std");
const vk = @import("vulkan");

pub fn VertexHashMap(comptime V: anytype) type {
    return std.HashMap(
        Vertex,
        V,
        VertexContext,
        std.hash_map.default_max_load_percentage,
    );
}

pub const VertexContext = struct {
    pub fn hash(self: VertexContext, k: Vertex) u64 {
        _ = self;
        return k.hash();
    }

    pub fn eql(self: VertexContext, s: Vertex, other: Vertex) bool {
        _ = self;
        return s.eql(other);
    }
};

const Vertex = @This();

pos: [3]f32,
color: [3]f32,
texCoord: [2]f32,

pub fn getBindingDescription() vk.VertexInputBindingDescription {
    return .{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };
}

pub fn getAttributeDescriptions() []const vk.VertexInputAttributeDescription {
    return &.{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "texCoord"),
        },
    };
}

fn hash(self: Vertex) u64 {
    var hasher = std.hash.Wyhash.init(1000);
    for (self.pos) |p| {
        const p1: u32 = @bitCast(p);
        hasher.update(std.mem.asBytes(&p1));
    }
    for (self.texCoord) |t| {
        const t1: u32 = @bitCast(t);
        hasher.update(std.mem.asBytes(&t1));
    }
    for (self.color) |c| {
        const c1: u32 = @bitCast(c);
        hasher.update(std.mem.asBytes(&c1));
    }
    return hasher.final();
}

fn eql(self: Vertex, other: Vertex) bool {
    return std.mem.eql(f32, &self.pos, &other.pos) and
        std.mem.eql(f32, &self.texCoord, &other.texCoord) and
        std.mem.eql(f32, &self.color, &other.color);
}
