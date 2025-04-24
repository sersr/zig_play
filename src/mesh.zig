const std = @import("std");
const vt = @import("vertex.zig");
const zm = @import("zmesh");

const tinyobj = @cImport({
    @cInclude("tinyobj_loader_c.h");
});

const Mesh = @This();
const Self = @This();

const IndexList = std.ArrayList(u32);
const VetexList = std.ArrayList(vt);

const Allocator = std.mem.Allocator;
allocator: Allocator,
indices: IndexList,
vertices: VetexList,

pub inline fn getVetexByteSize(self: Self) u64 {
    return @sizeOf(vt) * self.vertices.items.len;
}

pub inline fn getIndexByteSize(self: Self) u64 {
    return @sizeOf(u32) * self.indices.items.len;
}

pub fn init(allocator: Allocator) Self {
    return .{
        .allocator = allocator,
        .indices = .init(allocator),
        .vertices = .init(allocator),
    };
}

pub fn deinit(self: *Self) void {
    self.indices.deinit();
    self.vertices.deinit();
}

// 模型加载函数
pub fn loadModel(self: *Self, file_name: []const u8) !void {
    var attrib: tinyobj.tinyobj_attrib_t = undefined;
    var shapes: [*]tinyobj.tinyobj_shape_t = undefined;
    var materials: [*]tinyobj.tinyobj_material_t = undefined;
    var shapes_count: usize = 0;
    var materials_count: usize = 0;

    if (tinyobj.tinyobj_parse_obj(
        @ptrCast(&attrib),
        @ptrCast(&shapes),
        @ptrCast(&shapes_count),
        @ptrCast(&materials),
        @ptrCast(&materials_count),
        @ptrCast(file_name),
        loadObj,
        null,
        tinyobj.TINYOBJ_FLAG_TRIANGULATE,
    ) != 0) {
        return error.ModelLoadFailed;
    }

    var unique_vertices = vt.VertexHashMap(u32).init(self.allocator);
    defer unique_vertices.deinit();

    const face_verts = attrib.faces[0..attrib.num_faces];

    for (face_verts) |d| {
        const vert_index: usize = @intCast(d.v_idx);
        const tex_index: usize = @intCast(d.vt_idx);
        const vertex = vt{
            .pos = .{
                attrib.vertices[3 * vert_index],
                attrib.vertices[3 * vert_index + 1],
                attrib.vertices[3 * vert_index + 2],
            },
            .texCoord = .{
                attrib.texcoords[2 * tex_index],
                1.0 - attrib.texcoords[2 * tex_index + 1], // Y轴翻转
            },
            .color = .{ 1.0, 1.0, 1.0 },
        };

        const entry = try unique_vertices.getOrPut(vertex);
        if (!entry.found_existing) {
            entry.value_ptr.* = @intCast(self.vertices.items.len);
            try self.vertices.append(vertex);
        }
        try self.indices.append(entry.value_ptr.*);
    }
}

fn loadObj(
    ctx: ?*anyopaque,
    filename: [*c]const u8,
    is_mtl: c_int,
    obj_filename: [*c]const u8,
    buffer: [*c][*c]u8,
    len: [*c]usize,
) callconv(.c) void {
    _ = ctx;
    _ = is_mtl;
    _ = obj_filename;
    const filename_size = std.mem.len(filename);

    const file = std.fs.cwd().openFile(filename[0..filename_size], .{}) catch return;
    const meta = file.metadata() catch return;
    const size = meta.size();
    const al = std.heap.c_allocator;
    const bytes = file.readToEndAlloc(al, size) catch return;
    buffer.* = bytes.ptr;
    len.* = bytes.len;
}

pub const getBindingDescription = vt.getBindingDescription;
pub const getAttributeDescriptions = vt.getAttributeDescriptions;

fn getCwdPath(allocator: Allocator) ![]const u8 {
    return getCwdPathSub(allocator, "");
}
fn getCwdPathSub(allocator: Allocator, sub_path: []const u8) ![]const u8 {
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    const pa = try std.fs.cwd().realpath(sub_path, &path);

    const p_name = path[0 .. pa.len + 1];
    p_name[pa.len] = 0;
    const ret = try allocator.alloc(u8, p_name.len);

    @memcpy(ret, p_name);
    return ret;
}

pub fn loadModelGltf(self: *Self, file_name: []const u8) !void {
    zm.init(self.allocator);
    defer zm.deinit();

    const file_name_ab = try getCwdPathSub(self.allocator, file_name);
    const file_name_z: [:0]const u8 = @ptrCast(file_name_ab);

    const data = try zm.io.zcgltf.parseAndLoadFile(file_name_z);
    defer zm.io.zcgltf.freeData(data);

    var mesh_positions = std.ArrayList([3]f32).init(self.allocator);

    var tex_coords = std.ArrayList([2]f32).init(self.allocator);
    try zm.io.zcgltf.appendMeshPrimitive(
        data,
        0,
        0,
        &self.indices,
        &mesh_positions,
        null,
        &tex_coords,
        null,
    );

    std.debug.assert(mesh_positions.items.len == tex_coords.items.len);

    try self.vertices.resize(mesh_positions.items.len);
    for (self.vertices.items, mesh_positions.items, tex_coords.items) |*i, p, t| {
        i.* = .{ .pos = p, .color = .{ 1.0, 1.0, 1.0 }, .texCoord = t };
    }
}
