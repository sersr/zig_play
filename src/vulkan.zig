const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const zmath = @import("zmath");
const stbi = @import("stbi");
const Mesh = @import("mesh.zig");

const stb = @cImport({
    @cInclude("stb_image.h");
});

// const stb = @import("stb");

const tinyobj = @cImport({
    @cInclude("tinyobj_loader_c.h");
});

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const allocator = gpa.allocator();

const Self = @This();

window: *glfw.Window,
vkb: vk.BaseWrapper,
mesh: Mesh,

instance: vk.InstanceProxy = undefined,
// hide
_vk_inst_wrapper: vk.InstanceWrapper = undefined,

debugMessage: ?vk.DebugUtilsMessengerEXT = null,
vk_surface: glfw.VkSurfaceKHR = undefined,

// 物理设备
physical_device: vk.PhysicalDevice = .null_handle,
msaa_samples: vk.SampleCountFlags = .{ .@"1_bit" = true },
// 逻辑设配
// vk_device: ?vk.Device = null,
// hide
_device_wrapper: vk.DeviceWrapper = undefined,
device_instance: vk.DeviceProxy = undefined,

graphics_queue: vk.Queue = undefined,
present_queue: vk.Queue = undefined,

swapChain: vk.SwapchainKHR = undefined,
swaiChainImageFormat: vk.Format = undefined,
swapChainExtent: vk.Extent2D = undefined,

swapChainImages: VKImageArrType = .init(allocator),
swapchainImageViews: VKImageViewArrType = .init(allocator),
swapChainFramebuffers: VKFramebufferArrType = .init(allocator),

pipelineLayout: vk.PipelineLayout = undefined,
descriptorSetLayout: vk.DescriptorSetLayout = undefined,
renderPass: vk.RenderPass = undefined,
graphicsPipeline: vk.Pipeline = undefined,

commandPool: vk.CommandPool = undefined,

// color
color_image: vk.Image = undefined,
color_image_memory: vk.DeviceMemory = undefined,
color_image_view: vk.ImageView = undefined,

// depth
depth_image: vk.Image = undefined,
depth_image_memory: vk.DeviceMemory = undefined,
depth_image_view: vk.ImageView = undefined,

// smaple
mip_levels: u32 = undefined,
texture_image: vk.Image = undefined,
texture_image_memory: vk.DeviceMemory = undefined,
texture_image_view: vk.ImageView = undefined,
texture_sampler: vk.Sampler = undefined,

vertexBuffer: vk.Buffer = undefined,
vertexBufferMemory: vk.DeviceMemory = undefined,
indexBuffer: vk.Buffer = undefined,
indexBufferMemory: vk.DeviceMemory = undefined,

uniformBuffers: VKBufferArrType = .init(allocator),
uniformBufferMemory: VKDeviceMArrType = .init(allocator),
uniformBufferMapped: VKBufferMapped = .init(allocator),

descriptorPool: vk.DescriptorPool = undefined,
descriptorSets: VKDescSetArrType = .init(allocator),

commandBuffer: VKCmdbufferArrType = .init(allocator),

imageAvailableSemaphores: VKSemaphoreArrType = .init(allocator),
renderFinishedSemaphores: VKSemaphoreArrType = .init(allocator),
inFlightFence: VKFenceArrType = .init(allocator),
currentFrame: u32 = 0,
framebufferResize: bool = false,

const enableValidationLayers = builtin.mode == .Debug;
const VKImageArrType = std.ArrayList(vk.Image);
const VKImageViewArrType = std.ArrayList(vk.ImageView);
const VKFramebufferArrType = std.ArrayList(vk.Framebuffer);
const VKCmdbufferArrType = std.ArrayList(vk.CommandBuffer);
const VKSemaphoreArrType = std.ArrayList(vk.Semaphore);
const VKFenceArrType = std.ArrayList(vk.Fence);
const max_frames_in_flights = 2;
const VKBufferArrType = std.ArrayList(vk.Buffer);
const VKDeviceMArrType = std.ArrayList(vk.DeviceMemory);
const VKBufferMapped = std.ArrayList(*anyopaque);
const VKDescSetArrType = std.ArrayList(vk.DescriptorSet);

var init_once = std.once(init_global);

fn surface_enum(self: Self) vk.SurfaceKHR {
    return @enumFromInt(self.vk_surface);
}

pub fn init(width: i32, height: i32) !Self {
    init_once.call();

    try glfw.init();
    glfw.windowHint(glfw.ClientAPI, glfw.NoAPI);

    const window = try glfw.createWindow(width, height, "hello world", null, null);
    const vkb = vk.BaseWrapper.load(glfwGetInstanceProcAddress);
    return .{
        .window = window,
        .vkb = vkb,
        .mesh = .init(allocator),
    };
}

fn framebuferResizeCallback(window: ?*glfw.Window, width: c_int, height: c_int) callconv(.c) void {
    _ = width;
    _ = height;
    const p = glfw.getWindowUserPointer(window);
    const v: usize = @intFromPtr(p.?);
    const app: *Self = @ptrFromInt(v);
    app.framebufferResize = true;
}

pub fn deinit(self: Self) void {
    self.cleanupSwapChain();
    defer self.swapChainFramebuffers.deinit();
    defer self.swapchainImageViews.deinit();

    self.device_instance.destroyPipeline(self.graphicsPipeline, null);
    self.device_instance.destroyPipelineLayout(self.pipelineLayout, null);
    self.device_instance.destroyRenderPass(self.renderPass, null);

    for (self.uniformBuffers.items, self.uniformBufferMemory.items) |buffer, memory| {
        self.device_instance.destroyBuffer(buffer, null);
        self.device_instance.freeMemory(memory, null);
    }

    defer {
        self.uniformBufferMapped.deinit();
        self.uniformBuffers.deinit();
        self.uniformBufferMemory.deinit();
    }

    self.device_instance.destroyDescriptorPool(self.descriptorPool, null);
    self.device_instance.destroyDescriptorSetLayout(self.descriptorSetLayout, null);

    self.device_instance.destroyBuffer(self.indexBuffer, null);
    self.device_instance.freeMemory(self.indexBufferMemory, null);

    self.device_instance.destroyBuffer(self.vertexBuffer, null);
    self.device_instance.freeMemory(self.vertexBufferMemory, null);

    for (self.renderFinishedSemaphores.items, self.imageAvailableSemaphores.items, self.inFlightFence.items) |s1, s2, fence| {
        self.device_instance.destroySemaphore(s1, null);
        self.device_instance.destroySemaphore(s2, null);
        self.device_instance.destroyFence(fence, null);
    }

    defer {
        self.renderFinishedSemaphores.deinit();
        self.imageAvailableSemaphores.deinit();
        self.inFlightFence.deinit();
    }

    self.device_instance.destroyCommandPool(self.commandPool, null);

    defer self.swapChainImages.deinit();

    self.device_instance.destroyDevice(null);

    if (enableValidationLayers) {
        if (self.debugMessage) |e| {
            self.instance.destroyDebugUtilsMessengerEXT(e, null);
        }
    }

    self.instance.destroySurfaceKHR(self.surface_enum(), null);
    self.instance.destroyInstance(null);

    glfw.destroyWindow(self.window);

    glfw.terminate();
}

pub fn run(self: *Self) !void {
    glfw.setWindowUserPointer(self.window, self);
    _ = glfw.setFramebufferSizeCallback(self.window, framebuferResizeCallback);

    try self.initVulkan();

    std.log.info("run...", .{});

    while (!glfw.windowShouldClose(self.window)) {
        glfw.pollEvents();
        try self.drawFrame();
    }

    try self.device_instance.deviceWaitIdle();
}

fn debugCallback(
    message: vk.DebugUtilsMessageSeverityFlagsEXT,
    messageType: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(.c) vk.Bool32 {
    _ = message;
    _ = messageType;
    _ = p_user_data;

    if (p_callback_data) |data| {
        if (data.p_message) |m| {
            std.debug.print("validation layer: {s}\n", .{m});
        }
    }
    return vk.FALSE;
}

fn initVulkan(self: *Self) !void {
    const result = try self.checkValidationLayerSupport();
    if (!result) {
        std.log.info("error ", .{});
    }
    try self.createInstance();

    if (enableValidationLayers) {
        const createInfo = debugCreateInfo();

        if (self.instance.createDebugUtilsMessengerEXT(&createInfo, null)) |debug| {
            self.debugMessage = debug;
        } else |e| {
            std.log.err("debug error: {any}", .{e});
        }
    }

    self.createSurface();

    try self.pickPhysicalDevice();
    try self.createLogicalDevice();

    try self.createSwapChain();
    try self.createImageViews();

    try self.createRenderPass();
    try self.createDescriptorSetLayout();

    try self.createGraphicsPipeline();

    try self.createCommandPool();

    try self.createColorResources();
    try self.createDepthResources();
    try self.createFramebuffers();

    try self.createTextureImage();
    try self.createTextureImageView();
    try self.createTextureSampler();

    try self.loadModel();

    try self.createVertexBuffer();
    try self.createIndexBuffer();
    try self.createUniformBuffers();
    try self.createDescriptorPool();
    try self.createDescriptorSets();

    try self.createCommandBuffer();

    try self.createSyncObjects();
}

fn checkValidationLayerSupport(self: Self) !bool {
    const data = try self.vkb.enumerateInstanceLayerPropertiesAlloc(allocator);
    defer allocator.free(data);

    for (validationLayers) |layer| {
        var layerFound = false;
        const len = std.mem.len(layer);
        const layer_slice = layer[0..len];

        for (data) |item| {
            const v = std.mem.trim(u8, item.layer_name[0..], &[_]u8{0});
            if (std.mem.eql(u8, layer_slice, v)) {
                std.log.info("validation found: {s}, {s}", .{ layer_slice, item.layer_name });
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }

    return true;
}

fn debugCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
    return .{
        .message_severity = .{
            .error_bit_ext = true,
            // .info_bit_ext = true,
            .verbose_bit_ext = true,
            .warning_bit_ext = true,
        },
        .message_type = .{
            .general_bit_ext = true,
            .validation_bit_ext = true,
            .performance_bit_ext = true,
        },
        .pfn_user_callback = debugCallback,
    };
}

const validationLayers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
const deviceExtensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};
fn createInstance(self: *Self) !void {
    const vkb = self.vkb;
    const app_info: vk.ApplicationInfo = .{
        .p_application_name = "Hello world",
        .application_version = @bitCast(vk.makeApiVersion(0, 1, 0, 0)),
        .engine_version = @bitCast(vk.makeApiVersion(0, 1, 0, 0)),
        .api_version = @bitCast(vk.makeApiVersion(0, 1, 0, 0)),
        .p_engine_name = "no engine",
    };

    var createInfo: vk.InstanceCreateInfo = .{
        .p_application_info = &app_info,
        .flags = .{
            // fix macos error
            .enumerate_portability_bit_khr = true,
        },
    };
    var count: u32 = undefined;
    const exts = glfw.getRequiredInstanceExtensions(&count) orelse &[0][*:0]const u8{};
    const arrType = std.ArrayList([*:0]const u8);
    var arr = try arrType.initCapacity(allocator, count);
    defer arr.deinit();

    // fix macos error
    try arr.append(vk.extensions.khr_portability_enumeration.name);
    for (exts[0..count]) |ext| {
        try arr.append(ext);
    }

    if (enableValidationLayers) {
        try arr.append(vk.extensions.ext_debug_utils.name);
    }

    createInfo.enabled_extension_count = @intCast(arr.items.len);
    createInfo.pp_enabled_extension_names = arr.items.ptr;

    if (enableValidationLayers) {
        createInfo.pp_enabled_layer_names = &validationLayers;
        createInfo.enabled_layer_count = validationLayers.len;
        createInfo.p_next = &debugCreateInfo();
    } else {
        createInfo.enabled_layer_count = 0;
        createInfo.p_next = null;
    }

    const instance_handle = try vkb.createInstance(
        &createInfo,
        null,
    );

    self._vk_inst_wrapper = vk.InstanceWrapper.load(instance_handle, vkb.dispatch.vkGetInstanceProcAddr.?);
    self.instance = vk.InstanceProxy.init(instance_handle, &self._vk_inst_wrapper);
}

fn createSurface(self: *Self) void {
    const result = glfw.createWindowSurface(@intFromEnum(self.instance.handle), self.window, null, &self.vk_surface);
    print("create surface: {any}", .{result});
}

fn pickPhysicalDevice(self: *Self) !void {
    var device_count: u32 = 0;
    _ = try self.instance.enumeratePhysicalDevices(&device_count, null);

    const arrType = std.ArrayList(vk.PhysicalDevice);
    var arr = try arrType.initCapacity(allocator, device_count);
    defer arr.deinit();
    // 填充元素

    try arr.resize(device_count);

    _ = try self.instance.enumeratePhysicalDevices(&device_count, arr.items.ptr);
    print(".....len: {d}", .{device_count});

    for (arr.items) |item| {
        if (try self.isDeviceSuitable(item)) {
            self.physical_device = item;
            self.msaa_samples = self.getMaxUsableSampleCount();
            print("physical device: {any}, {any}", .{ item, self.msaa_samples });
            break;
        }
    }

    if (self.physical_device == .null_handle) {
        @panic("failed to find a suitable GPU!");
    }
}

fn createLogicalDevice(self: *Self) !void {
    const indices = try self.findQueueFamilies(self.physical_device);

    const h = std.AutoHashMap(u32, void);
    var uniqueQueueFamilies = h.init(allocator);
    try uniqueQueueFamilies.put(indices.graphicsFaimily.?, {});
    try uniqueQueueFamilies.put(indices.presentFamily.?, {});

    var it = uniqueQueueFamilies.keyIterator();
    const DevType = std.ArrayList(vk.DeviceQueueCreateInfo);
    var dev_arr = DevType.init(allocator);

    const queuePriority = [_]f32{1.0};

    while (it.next()) |item| {
        try dev_arr.append(vk.DeviceQueueCreateInfo{
            .s_type = .device_queue_create_info,
            .queue_family_index = item.*,
            .queue_count = 1,
            .p_queue_priorities = &queuePriority,
        });
    }

    const deviceFeatures: vk.PhysicalDeviceFeatures =
        self.instance.getPhysicalDeviceFeatures(self.physical_device);

    var createInfo: vk.DeviceCreateInfo = .{
        .p_queue_create_infos = dev_arr.items.ptr,
        .queue_create_info_count = @intCast(dev_arr.items.len),
        .p_enabled_features = &deviceFeatures,
        .enabled_extension_count = @intCast(deviceExtensions.len),
        .pp_enabled_extension_names = &deviceExtensions,
    };

    if (enableValidationLayers) {
        createInfo.enabled_layer_count = @intCast(validationLayers.len);
        createInfo.pp_enabled_layer_names = &validationLayers;
    } else {
        createInfo.enabled_layer_count = 0;
    }

    const vk_device = try self.instance.createDevice(self.physical_device, &createInfo, null);

    self._device_wrapper = vk.DeviceWrapper.load(vk_device, self.instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
    self.device_instance = vk.DeviceProxy.init(vk_device, &self._device_wrapper);

    self.graphics_queue = self.device_instance.getDeviceQueue(indices.graphicsFaimily.?, 0);
    self.present_queue = self.device_instance.getDeviceQueue(indices.presentFamily.?, 0);
}

fn createSwapChain(self: *Self) !void {
    var swap_chain_support = try self.querySwapChainSupport(self.physical_device);
    defer swap_chain_support.deinit();

    const surfaceFormat = chooseSwapSurfaceFormat(swap_chain_support.formats.?.items);
    const presentMode = chooseSwapPresentMode(swap_chain_support.presentModes.?.items);
    const extent = self.chooseSwapExtent(&swap_chain_support.capabilities);

    var imageCount: u32 = swap_chain_support.capabilities.min_image_count + 1;
    if (swap_chain_support.capabilities.max_image_count > 0 and imageCount > swap_chain_support.capabilities.max_image_count) {
        imageCount = swap_chain_support.capabilities.max_image_count;
    }

    var createInfo: vk.SwapchainCreateInfoKHR = .{
        .image_sharing_mode = .concurrent,
        .pre_transform = swap_chain_support.capabilities.current_transform,
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = presentMode,
        .clipped = vk.TRUE,
        .surface = self.surface_enum(),
        .min_image_count = imageCount,
        .image_format = surfaceFormat.format,
        .image_color_space = surfaceFormat.color_space,
        .image_extent = extent,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
    };

    const indices = try self.findQueueFamilies(self.physical_device);

    const queueFamilyIndices = [_]u32{ indices.graphicsFaimily.?, indices.presentFamily.? };

    if (indices.graphicsFaimily.? != indices.presentFamily.?) {
        createInfo.image_sharing_mode = .concurrent;
        createInfo.queue_family_index_count = 2;
        createInfo.p_queue_family_indices = &queueFamilyIndices;
    } else {
        createInfo.image_sharing_mode = .exclusive;
    }

    createInfo.old_swapchain = .null_handle;

    self.swapChain = try self.device_instance.createSwapchainKHR(&createInfo, null);

    _ = try self.device_instance.getSwapchainImagesKHR(self.swapChain, &imageCount, null);
    try self.swapChainImages.resize(imageCount);
    _ = try self.device_instance.getSwapchainImagesKHR(self.swapChain, &imageCount, self.swapChainImages.items.ptr);

    self.swaiChainImageFormat = surfaceFormat.format;
    self.swapChainExtent = extent;
    print("extent: {any}", .{extent});
}

fn createImageViews(self: *Self) !void {
    try self.swapchainImageViews.resize(self.swapChainImages.items.len);

    for (self.swapChainImages.items, self.swapchainImageViews.items) |image, *view| {
        var createInfo: vk.ImageViewCreateInfo = .{
            .image = image,
            .view_type = .@"2d",
            .format = self.swaiChainImageFormat,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };

        view.* = try self.device_instance.createImageView(&createInfo, null);
    }
}

fn createRenderPass(self: *Self) !void {
    const colorAttachment: vk.AttachmentDescription = .{
        .format = self.swaiChainImageFormat,
        .samples = self.msaa_samples,
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .color_attachment_optimal,
    };

    const depthAttachment: vk.AttachmentDescription = .{
        .format = try self.findDepthFormat(),
        .samples = self.msaa_samples,
        .load_op = .clear,
        .store_op = .dont_care,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };

    const colorAttachmentResolve: vk.AttachmentDescription = .{
        .format = self.swaiChainImageFormat,
        .samples = .{ .@"1_bit" = true },
        .load_op = .dont_care,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const colorAttchmentRef: vk.AttachmentReference = .{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const depthAttachmentRef: vk.AttachmentReference = .{
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };
    const colorAttachmentResolveRef: vk.AttachmentReference = .{
        .attachment = 2,
        .layout = .color_attachment_optimal,
    };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&colorAttchmentRef),
        .p_depth_stencil_attachment = @ptrCast(&depthAttachmentRef),
        .p_resolve_attachments = @ptrCast(&colorAttachmentResolveRef),
    };

    const dependency: vk.SubpassDependency = .{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
    };

    const attachements = [_]vk.AttachmentDescription{ colorAttachment, depthAttachment, colorAttachmentResolve };

    const renderPassInfo: vk.RenderPassCreateInfo = .{
        .attachment_count = attachements.len,
        .p_attachments = @ptrCast(&attachements),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    };

    self.renderPass = try self.device_instance.createRenderPass(&renderPassInfo, null);
}

fn createDescriptorSetLayout(self: *Self) !void {
    const uboLayoutBinding: vk.DescriptorSetLayoutBinding = .{
        .binding = 0,
        .descriptor_count = 1,
        .descriptor_type = .uniform_buffer,
        .p_immutable_samplers = null,
        .stage_flags = .{ .vertex_bit = true },
    };

    const sampleLayoutBinding: vk.DescriptorSetLayoutBinding = .{
        .binding = 1,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_immutable_samplers = null,
        .stage_flags = .{ .fragment_bit = true },
    };

    const binding = [_]vk.DescriptorSetLayoutBinding{
        uboLayoutBinding, sampleLayoutBinding,
    };

    const layoutInfo: vk.DescriptorSetLayoutCreateInfo = .{
        .binding_count = binding.len,
        .p_bindings = @ptrCast(&binding),
    };

    self.descriptorSetLayout = try self.device_instance.createDescriptorSetLayout(&layoutInfo, null);
}

fn createGraphicsPipeline(self: *Self) !void {
    const vert_shader = try readFile("shaders/vert.spv");
    const frag_shader = try readFile("shaders/frag.spv");

    const verShaderModule = try self.createShaderModule(vert_shader);
    const fragShaderModule = try self.createShaderModule(frag_shader);

    const verShaderStageInfo: vk.PipelineShaderStageCreateInfo = .{
        .stage = .{ .vertex_bit = true },
        .module = verShaderModule,
        .p_name = "main",
    };

    const fragShaderStageInfo: vk.PipelineShaderStageCreateInfo = .{
        .stage = .{ .fragment_bit = true },
        .module = fragShaderModule,
        .p_name = "main",
    };

    const shaderStages = [_]vk.PipelineShaderStageCreateInfo{
        verShaderStageInfo,
        fragShaderStageInfo,
    };

    const bindingDesc = Mesh.getBindingDescription();
    const attributeDesc = Mesh.getAttributeDescriptions();
    const vertexInputInfo: vk.PipelineVertexInputStateCreateInfo = .{
        .vertex_binding_description_count = 1,
        .vertex_attribute_description_count = @intCast(attributeDesc.len),
        .p_vertex_binding_descriptions = @ptrCast(&bindingDesc),
        .p_vertex_attribute_descriptions = attributeDesc.ptr,
    };

    const inputAssembly: vk.PipelineInputAssemblyStateCreateInfo = .{
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const viewportState: vk.PipelineViewportStateCreateInfo = .{
        .viewport_count = 1,
        .scissor_count = 1,
    };

    const rasterizer: vk.PipelineRasterizationStateCreateInfo = .{
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .line_width = 1.0,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0.0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
    };

    const mulitsampling: vk.PipelineMultisampleStateCreateInfo = .{
        .sample_shading_enable = vk.FALSE,
        .rasterization_samples = self.msaa_samples,
        .alpha_to_coverage_enable = 0,
        .alpha_to_one_enable = 0,
        .min_sample_shading = 0,
    };

    const depthStencil: vk.PipelineDepthStencilStateCreateInfo = .{
        .depth_test_enable = vk.TRUE,
        .depth_write_enable = vk.TRUE,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = vk.FALSE,
        .stencil_test_enable = vk.FALSE,
        .back = std.mem.zeroes(vk.StencilOpState),
        .front = std.mem.zeroes(vk.StencilOpState),
        .max_depth_bounds = 0.0,
        .min_depth_bounds = 0.0,
    };

    const clolorBlendAttachment: vk.PipelineColorBlendAttachmentState = .{
        .blend_enable = vk.FALSE,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .zero,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .src_color_blend_factor = .zero,
    };

    const colorBlending: vk.PipelineColorBlendStateCreateInfo = .{
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&clolorBlendAttachment),
        .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    const dynamicStates = [_]vk.DynamicState{ vk.DynamicState.viewport, vk.DynamicState.scissor };

    const dynamicState: vk.PipelineDynamicStateCreateInfo = .{
        .dynamic_state_count = dynamicStates.len,
        .p_dynamic_states = @ptrCast(&dynamicStates),
    };

    const pipelineLayoutInfo: vk.PipelineLayoutCreateInfo = .{
        .set_layout_count = 1,
        .push_constant_range_count = 0,
        .p_set_layouts = @ptrCast(&self.descriptorSetLayout),
    };

    self.pipelineLayout = try self.device_instance.createPipelineLayout(&pipelineLayoutInfo, null);

    const pipelineInfo: vk.GraphicsPipelineCreateInfo = .{
        .stage_count = 2,
        .base_pipeline_index = 0,
        .p_stages = @ptrCast(&shaderStages),
        .p_vertex_input_state = &vertexInputInfo,
        .p_input_assembly_state = &inputAssembly,
        .p_viewport_state = &viewportState,
        .p_rasterization_state = &rasterizer,
        .p_multisample_state = &mulitsampling,
        .p_depth_stencil_state = &depthStencil,
        .p_color_blend_state = &colorBlending,
        .p_dynamic_state = &dynamicState,
        .layout = self.pipelineLayout,
        .render_pass = self.renderPass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
    };

    _ = try self.device_instance.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&pipelineInfo),
        null,
        @ptrCast(&self.graphicsPipeline),
    );

    self.device_instance.destroyShaderModule(verShaderModule, null);
    self.device_instance.destroyShaderModule(fragShaderModule, null);
}

fn createFramebuffers(self: *Self) !void {
    try self.swapChainFramebuffers.resize(self.swapchainImageViews.items.len);

    for (self.swapchainImageViews.items, self.swapChainFramebuffers.items) |item, *buf| {
        const attachments = [_]vk.ImageView{
            self.color_image_view,
            self.depth_image_view,
            item,
        };
        const framebufferInfo: vk.FramebufferCreateInfo = .{
            .render_pass = self.renderPass,
            .attachment_count = attachments.len,
            .p_attachments = @ptrCast(&attachments),
            .width = self.swapChainExtent.width,
            .height = self.swapChainExtent.height,
            .layers = 1,
        };

        buf.* = try self.device_instance.createFramebuffer(&framebufferInfo, null);
    }
}

fn createCommandPool(self: *Self) !void {
    const queueFamilyIndices = try self.findQueueFamilies(self.physical_device);
    const poolInfo: vk.CommandPoolCreateInfo = .{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queueFamilyIndices.graphicsFaimily.?,
    };

    self.commandPool = try self.device_instance.createCommandPool(&poolInfo, null);
}

// image

// 颜色资源创建函数
fn createColorResources(self: *Self) !void {
    const color_format = self.swaiChainImageFormat;
    const samples = self.msaa_samples;

    // 创建多采样颜色图像
    try self.createImage(
        self.swapChainExtent.width,
        self.swapChainExtent.height,
        1, // mipLevels
        samples,
        color_format,
        .optimal,
        .{
            .transient_attachment_bit = true,
            .color_attachment_bit = true,
        },
        .{ .device_local_bit = true },
        &self.color_image,
        &self.color_image_memory,
    );

    // 创建颜色图像视图
    self.color_image_view = try self.createImageView(self.color_image, color_format, .{ .color_bit = true }, 1);
}

// 深度资源创建函数
fn createDepthResources(self: *Self) !void {
    const depth_format = try self.findDepthFormat();
    const samples = self.msaa_samples;

    // 创建深度图像
    try self.createImage(self.swapChainExtent.width, self.swapChainExtent.height, 1, // mipLevels
        samples, depth_format, .optimal, .{ .depth_stencil_attachment_bit = true }, .{ .device_local_bit = true }, &self.depth_image, &self.depth_image_memory);

    // 创建深度图像视图
    self.depth_image_view = try self.createImageView(self.depth_image, depth_format, .{ .depth_bit = true }, // 如果格式包含模板则添加.stencil_bit
        1);
}

// 支持格式查找函数
fn findSupportedFormat(self: *Self, candidates: []const vk.Format, tiling: vk.ImageTiling, features: vk.FormatFeatureFlags) !vk.Format {
    for (candidates) |format| {
        const props = self.instance.getPhysicalDeviceFormatProperties(self.physical_device, format);

        const check_features = switch (tiling) {
            .linear => props.linear_tiling_features,
            .optimal => props.optimal_tiling_features,
            else => unreachable,
        };

        if (check_features.contains(features)) {
            return format;
        }
    }
    return error.FormatNotSupported;
}

// 深度格式查找函数
fn findDepthFormat(self: *Self) !vk.Format {
    const candidates = [_]vk.Format{ .d32_sfloat, .d32_sfloat_s8_uint, .d24_unorm_s8_uint };

    return try self.findSupportedFormat(&candidates, .optimal, .{ .depth_stencil_attachment_bit = true });
}
fn hasStencilComponent(format: vk.Format) bool {
    return format == .d32_sfloat_s8_uint or format == .d24_unorm_s8_uint;
}

// 纹理图像创建函数
fn createTextureImage(self: *Self) !void {
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    const p: []u8 = (&path);
    const pa = try std.fs.cwd().realpath(TEXTURE_PATH, p);
    print("path: {s}", .{pa});
    path[pa.len] = 0;

    stbi.init(std.heap.c_allocator);

    // 加载纹理数据
    var tex_width: c_int = undefined;
    var tex_height: c_int = undefined;
    var tex_channels: c_int = undefined;
    const pixels = stb.stbi_load(@ptrCast(&path), &tex_width, &tex_height, &tex_channels, stb.STBI_rgb_alpha);

    if (pixels == null) return error.TextureLoadFailed;
    defer stb.stbi_image_free(pixels);

    const image_size = @as(vk.DeviceSize, @intCast(tex_width * tex_height * 4));
    self.mip_levels = @as(u32, @intFromFloat(std.math.log2(@as(f32, @floatFromInt(@max(tex_width, tex_height)))))) + 1;

    // 创建暂存缓冲
    var staging_buffer: vk.Buffer = undefined;
    var staging_memory: vk.DeviceMemory = undefined;
    try self.createBuffer(image_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, &staging_buffer, &staging_memory);
    defer {
        self.device_instance.destroyBuffer(staging_buffer, null);
        self.device_instance.freeMemory(staging_memory, null);
    }

    // 映射内存并拷贝数据
    const data = try self.device_instance.mapMemory(staging_memory, 0, image_size, .{});
    @memcpy(@as([*]u8, @ptrCast(data))[0..image_size], @as([*]const u8, @ptrCast(pixels))[0..image_size]);
    print("....sss {any}", .{data.?});
    self.device_instance.unmapMemory(staging_memory);
    print("....sss {any}", .{data.?});
    // 创建设备本地图像
    try self.createImage(
        @intCast(tex_width),
        @intCast(tex_height),
        self.mip_levels,
        .{ .@"1_bit" = true },
        .r8g8b8a8_srgb,
        .optimal,
        .{ .transfer_src_bit = true, .transfer_dst_bit = true, .sampled_bit = true },
        .{ .device_local_bit = true },
        &self.texture_image,
        &self.texture_image_memory,
    );

    // 转换图像布局并拷贝数据
    try self.transitionImageLayout(self.texture_image, .r8g8b8a8_srgb, .undefined, .transfer_dst_optimal, self.mip_levels);
    try self.copyBufferToImage(staging_buffer, self.texture_image, @intCast(tex_width), @intCast(tex_height));

    // 生成Mipmap链
    try self.generateMipmaps(self.texture_image, .r8g8b8a8_srgb, tex_width, tex_height, self.mip_levels);
}

// Mipmap生成函数
fn generateMipmaps(self: *Self, image: vk.Image, image_format: vk.Format, tex_width: i32, tex_height: i32, mip_levels: u32) !void {
    // 检查格式支持
    const format_props =
        self.instance.getPhysicalDeviceFormatProperties(self.physical_device, image_format);

    if (!format_props.optimal_tiling_features.sampled_image_filter_linear_bit) {
        return error.TextureFormatNotSupported;
    }

    const cmd_buffer = try self.beginSingleTimeCommands();

    var barrier = vk.ImageMemoryBarrier{
        .s_type = .image_memory_barrier,
        .p_next = null,
        .src_access_mask = .{},
        .dst_access_mask = .{},
        .old_layout = .undefined,
        .new_layout = .undefined,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    };

    var mip_width = tex_width;
    var mip_height = tex_height;

    for (1..mip_levels) |i| {
        // 源层级屏障
        barrier.subresource_range.base_mip_level = @intCast(i - 1);
        barrier.old_layout = .transfer_dst_optimal;
        barrier.new_layout = .transfer_src_optimal;
        barrier.src_access_mask = .{ .transfer_write_bit = true };
        barrier.dst_access_mask = .{ .transfer_read_bit = true };

        self.device_instance.cmdPipelineBarrier(cmd_buffer, .{ .transfer_bit = true }, .{ .transfer_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&barrier));

        // 图像blit操作
        const blit = vk.ImageBlit{
            .src_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = mip_width, .y = mip_height, .z = 1 },
            },
            .src_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = @intCast(i - 1),
                .base_array_layer = 0,
                .layer_count = 1,
            },
            .dst_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = if (mip_width > 1) @divFloor(mip_width, 2) else 1, .y = if (mip_height > 1) @divFloor(mip_height, 2) else 1, .z = 1 },
            },
            .dst_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .mip_level = @intCast(i),
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };

        self.device_instance.cmdBlitImage(cmd_buffer, image, .transfer_src_optimal, image, .transfer_dst_optimal, 1, @ptrCast(&blit), .linear);

        // 转换到shader只读布局
        barrier.old_layout = .transfer_src_optimal;
        barrier.new_layout = .shader_read_only_optimal;
        barrier.src_access_mask = .{ .transfer_read_bit = true };
        barrier.dst_access_mask = .{ .shader_read_bit = true };

        self.device_instance.cmdPipelineBarrier(cmd_buffer, .{ .transfer_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&barrier));

        mip_width = if (mip_width > 1) @divFloor(mip_width, 2) else mip_width;
        mip_height = if (mip_height > 1) @divFloor(mip_height, 2) else mip_height;
    }

    // 处理最后一个mip层级
    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = .transfer_dst_optimal;
    barrier.new_layout = .shader_read_only_optimal;
    barrier.src_access_mask = .{ .transfer_write_bit = true };
    barrier.dst_access_mask = .{ .shader_read_bit = true };

    self.device_instance.cmdPipelineBarrier(cmd_buffer, .{ .transfer_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&barrier));

    try self.endSingleTimeCommands(cmd_buffer);
}

// 获取最大采样数
fn getMaxUsableSampleCount(self: *Self) vk.SampleCountFlags {
    var props =
        self.instance.getPhysicalDeviceProperties(self.physical_device);

    const counts = props.limits.framebuffer_color_sample_counts.intersect(props.limits.framebuffer_depth_sample_counts);

    if (counts.@"64_bit") {
        return .{ .@"64_bit" = true };
    } else if (counts.@"32_bit") {
        return .{ .@"32_bit" = true };
    } else if (counts.@"16_bit") {
        return .{ .@"16_bit" = true };
    } else if (counts.@"8_bit") {
        return .{ .@"8_bit" = true };
    } else if (counts.@"4_bit") {
        return .{ .@"4_bit" = true };
    } else if (counts.@"2_bit") {
        return .{ .@"2_bit" = true };
    }

    return .{ .@"1_bit" = true };
}
fn createTextureImageView(self: *Self) !void {
    self.texture_image_view = try self.createImageView(self.texture_image, vk.Format.r8g8b8a8_srgb, .{ .color_bit = true }, self.mip_levels);
}

// 纹理采样器创建函数
fn createTextureSampler(self: *Self) !void {
    const properties = self.instance.getPhysicalDeviceProperties(self.physical_device);

    const sampler_info = vk.SamplerCreateInfo{
        .s_type = .sampler_create_info,
        .mag_filter = .linear,
        .min_filter = .linear,
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
        .anisotropy_enable = vk.TRUE,
        .max_anisotropy = properties.limits.max_sampler_anisotropy,
        .border_color = .int_opaque_black,
        .unnormalized_coordinates = vk.FALSE,
        .compare_enable = vk.FALSE,
        .compare_op = .always,
        .mipmap_mode = .linear,
        .min_lod = 0.0,
        .max_lod = vk.LOD_CLAMP_NONE,
        .mip_lod_bias = 0.0,
        .p_next = null,
        .flags = .{},
    };

    self.texture_sampler = try self.device_instance.createSampler(&sampler_info, null);
}

// 图像视图创建函数
fn createImageView(self: *Self, image: vk.Image, format: vk.Format, aspect_flags: vk.ImageAspectFlags, mip_levels: u32) !vk.ImageView {
    const view_info = vk.ImageViewCreateInfo{
        .s_type = .image_view_create_info,
        .image = image,
        .view_type = .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{
            .aspect_mask = aspect_flags,
            .base_mip_level = 0,
            .level_count = mip_levels,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .p_next = null,
        .flags = .{},
    };

    return try self.device_instance.createImageView(&view_info, null);
}

// 图像创建函数
fn createImage(self: *Self, width: u32, height: u32, mip_levels: u32, num_samples: vk.SampleCountFlags, format: vk.Format, tiling: vk.ImageTiling, usage: vk.ImageUsageFlags, properties: vk.MemoryPropertyFlags, image: *vk.Image, image_memory: *vk.DeviceMemory) !void {
    // 1. 初始化图像创建信息 (参考网页2、7)
    const image_info = vk.ImageCreateInfo{
        .s_type = .image_create_info,
        .image_type = .@"2d",
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = mip_levels,
        .array_layers = 1,
        .format = format,
        .tiling = tiling,
        .initial_layout = .undefined,
        .usage = usage,
        .samples = num_samples,
        .sharing_mode = .exclusive,
        // 其他字段初始化为默认值
        .flags = .{},
        .p_next = null,
        .queue_family_index_count = 0,
        .p_queue_family_indices = null,
    };

    // 2. 创建图像对象 (参考网页6、7)
    image.* = try self.device_instance.createImage(&image_info, null);

    // 3. 获取内存需求 (参考网页8)
    const mem_requirements = self.device_instance.getImageMemoryRequirements(image.*);

    // 4. 分配内存 (参考网页8、9)
    const alloc_info = vk.MemoryAllocateInfo{
        .s_type = .memory_allocate_info,
        .allocation_size = mem_requirements.size,
        .memory_type_index = self.findMemoryType(mem_requirements.memory_type_bits, properties),
        .p_next = null,
    };
    image_memory.* = try self.device_instance.allocateMemory(&alloc_info, null);

    // 5. 绑定内存到图像 (参考网页6、8)
    try self.device_instance.bindImageMemory(image.*, image_memory.*, 0);
}
// 图像布局转换函数
fn transitionImageLayout(self: *Self, image: vk.Image, format: vk.Format, old_layout: vk.ImageLayout, new_layout: vk.ImageLayout, mip_levels: u32) !void {
    _ = format;
    const cmd_buffer = try self.beginSingleTimeCommands();
    defer {
        self.endSingleTimeCommands(cmd_buffer) catch {};
    }

    var barrier = vk.ImageMemoryBarrier{
        .s_type = .image_memory_barrier,
        .src_access_mask = .{},
        .dst_access_mask = .{},
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = mip_levels,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .p_next = null,
    };

    var src_stage: vk.PipelineStageFlags = .{};
    var dst_stage: vk.PipelineStageFlags = .{};

    // 布局转换条件判断
    if (old_layout == .undefined and new_layout == .transfer_dst_optimal) {
        barrier.src_access_mask = .{};
        barrier.dst_access_mask = .{ .transfer_write_bit = true };
        src_stage = .{ .top_of_pipe_bit = true };
        dst_stage = .{ .transfer_bit = true };
    } else if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) {
        barrier.src_access_mask = .{ .transfer_write_bit = true };
        barrier.dst_access_mask = .{ .shader_read_bit = true };
        src_stage = .{ .transfer_bit = true };
        dst_stage = .{ .fragment_shader_bit = true };
    } else {
        return error.UnsupportedLayoutTransition;
    }

    self.device_instance.cmdPipelineBarrier(cmd_buffer, src_stage, dst_stage, .{}, 0, null, 0, null, 1, @ptrCast(&barrier));
}
// 缓冲到图像拷贝函数
fn copyBufferToImage(self: *Self, buffer: vk.Buffer, image: vk.Image, width: u32, height: u32) !void {
    const cmd_buffer = try self.beginSingleTimeCommands();
    defer {
        self.endSingleTimeCommands(cmd_buffer) catch {};
    }

    const region = vk.BufferImageCopy{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = width, .height = height, .depth = 1 },
    };

    self.device_instance.cmdCopyBufferToImage(cmd_buffer, buffer, image, .transfer_dst_optimal, 1, @ptrCast(&region));
}

const MODEL_PATH = "resources/viking_room.obj";
const TEXTURE_PATH = "resources/viking_room.png";

fn loadModel(self: *Self) !void {
    // try self.mesh.loadModel(@ptrCast(MODEL_PATH));
    try self.mesh.loadModelGltf("resources/cube.gltf");
}
// end image

fn createVertexBuffer(self: *Self) !void {
    const bufferSize: vk.DeviceSize = self.mesh.getVetexByteSize();

    var stagingBuffer: vk.Buffer = undefined;
    var stagingBufferMemory: vk.DeviceMemory = undefined;
    try self.createBuffer(
        bufferSize,
        .{
            .transfer_src_bit = true,
        },
        .{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        },
        &stagingBuffer,
        &stagingBufferMemory,
    );
    const data = try self.device_instance.mapMemory(stagingBufferMemory, 0, bufferSize, .fromInt(0));

    if (data == null) {
        @panic("failed to map memory!");
    }
    const dst_p: [*]u8 = @ptrCast(data.?);
    const src: []const u8 = @ptrCast(self.mesh.vertices.items);
    @memcpy(dst_p, src);

    self.device_instance.unmapMemory(stagingBufferMemory);

    try self.createBuffer(
        bufferSize,
        .{
            .transfer_dst_bit = true,
            .vertex_buffer_bit = true,
        },
        .{ .device_local_bit = true },
        &self.vertexBuffer,
        &self.vertexBufferMemory,
    );

    try self.copyBuffer(stagingBuffer, self.vertexBuffer, bufferSize);

    self.device_instance.destroyBuffer(stagingBuffer, null);
    self.device_instance.freeMemory(stagingBufferMemory, null);
}

fn createIndexBuffer(self: *Self) !void {
    const bufferSize: vk.DeviceSize = self.mesh.getIndexByteSize();

    var stagingBuffer: vk.Buffer = undefined;
    var stagingBufferMemory: vk.DeviceMemory = undefined;
    try self.createBuffer(
        bufferSize,
        .{
            .transfer_src_bit = true,
        },
        .{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        },
        &stagingBuffer,
        &stagingBufferMemory,
    );
    const data = try self.device_instance.mapMemory(stagingBufferMemory, 0, bufferSize, .fromInt(0));

    if (data == null) {
        @panic("failed to map memory!");
    }

    const dst_p: [*]u8 = @ptrCast(data.?);
    const src: []const u8 = @ptrCast(self.mesh.indices.items);
    @memcpy(dst_p, src);

    self.device_instance.unmapMemory(stagingBufferMemory);

    try self.createBuffer(
        bufferSize,
        .{
            .transfer_dst_bit = true,
            .index_buffer_bit = true,
        },
        .{ .device_local_bit = true },
        &self.indexBuffer,
        &self.indexBufferMemory,
    );

    try self.copyBuffer(stagingBuffer, self.indexBuffer, bufferSize);

    self.device_instance.destroyBuffer(stagingBuffer, null);
    self.device_instance.freeMemory(stagingBufferMemory, null);
}

fn createUniformBuffers(self: *Self) !void {
    const bufferSize = @sizeOf(UniformBufferObject);
    try self.uniformBuffers.resize(max_frames_in_flights);
    try self.uniformBufferMemory.resize(max_frames_in_flights);
    try self.uniformBufferMapped.resize(max_frames_in_flights);

    for (self.uniformBuffers.items, self.uniformBufferMemory.items, self.uniformBufferMapped.items) |*buffer, *m, *mapped| {
        try self.createBuffer(
            bufferSize,
            .{ .uniform_buffer_bit = true },
            .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            },
            buffer,
            m,
        );
        mapped.* = (try self.device_instance.mapMemory(m.*, 0, bufferSize, .fromInt(0))).?;
    }
}

fn createBuffer(self: Self, size: vk.DeviceSize, usage: vk.BufferUsageFlags, properties: vk.MemoryPropertyFlags, buffer: *vk.Buffer, bufferMemory: *vk.DeviceMemory) !void {
    const bufferInfo: vk.BufferCreateInfo = .{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    };

    buffer.* = try self.device_instance.createBuffer(&bufferInfo, null);

    const memRequirements = self.device_instance.getBufferMemoryRequirements(buffer.*);

    const alocInfo: vk.MemoryAllocateInfo = .{
        .allocation_size = memRequirements.size,
        .memory_type_index = self.findMemoryType(memRequirements.memory_type_bits, properties),
    };

    bufferMemory.* = try self.device_instance.allocateMemory(&alocInfo, null);

    try self.device_instance.bindBufferMemory(buffer.*, bufferMemory.*, 0);
}

fn copyBuffer(self: *Self, srcBuffer: vk.Buffer, dstBuffer: vk.Buffer, size: vk.DeviceSize) !void {
    const allocInfo: vk.CommandBufferAllocateInfo = .{
        .level = .primary,
        .command_pool = self.commandPool,
        .command_buffer_count = 1,
    };

    var commandBuffer: vk.CommandBuffer = undefined;
    _ = try self.device_instance.allocateCommandBuffers(&allocInfo, @ptrCast(&commandBuffer));

    const beginInfo: vk.CommandBufferBeginInfo = .{
        .flags = .{ .one_time_submit_bit = true },
    };

    try self.device_instance.beginCommandBuffer(commandBuffer, &beginInfo);

    const copyRegin: vk.BufferCopy = .{
        .size = size,
        .dst_offset = 0,
        .src_offset = 0,
    };
    self.device_instance.cmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, @ptrCast(&copyRegin));

    try self.device_instance.endCommandBuffer(commandBuffer);

    const submitInfo: vk.SubmitInfo = .{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&commandBuffer),
    };

    try self.device_instance.queueSubmit(self.graphics_queue, 1, @ptrCast(&submitInfo), .null_handle);
    try self.device_instance.queueWaitIdle(self.graphics_queue);
    self.device_instance.freeCommandBuffers(self.commandPool, 1, @ptrCast(&commandBuffer));
}

fn findMemoryType(self: Self, typeFilter: u32, properties: vk.MemoryPropertyFlags) u32 {
    const memProperties = self.instance.getPhysicalDeviceMemoryProperties(self.physical_device);

    const vi: u32 = 1;
    for (0..memProperties.memory_type_count) |index| {
        const item = memProperties.memory_types[index];
        const i: u5 = @intCast(index);
        const v = vi << i;
        if ((typeFilter & v != 0) and (item.property_flags.toInt() & properties.toInt()) == properties.toInt()) {
            return i;
        }
    }

    @panic("failed to find suitable memory type!");
}
fn createCommandBuffer(self: *Self) !void {
    try self.commandBuffer.resize(max_frames_in_flights);

    const allocInfo: vk.CommandBufferAllocateInfo = .{
        .command_pool = self.commandPool,
        .level = .primary,
        .command_buffer_count = @intCast(self.commandBuffer.items.len),
    };

    try self.device_instance.allocateCommandBuffers(
        &allocInfo,
        self.commandBuffer.items.ptr,
    );
}

fn recordCommandBuffer(self: *Self, commandBuffer: vk.CommandBuffer, imageIndex: u32) !void {
    const beginInfo: vk.CommandBufferBeginInfo = .{};

    try self.device_instance.beginCommandBuffer(commandBuffer, &beginInfo);

    var renderPassInfo: vk.RenderPassBeginInfo = .{
        .render_pass = self.renderPass,
        .framebuffer = self.swapChainFramebuffers.items[imageIndex],
        .render_area = .{ .extent = self.swapChainExtent, .offset = .{ .x = 0, .y = 0 } },
    };

    const clearColors = [_]vk.ClearValue{ .{ .color = .{
        .float_32 = .{ 0.0, 0.0, 0.0, 0.0 },
    } }, .{ .depth_stencil = .{ .depth = 1.0, .stencil = 0 } } };

    renderPassInfo.clear_value_count = clearColors.len;
    renderPassInfo.p_clear_values = @ptrCast(&clearColors);

    self.device_instance.cmdBeginRenderPass(commandBuffer, &renderPassInfo, .@"inline");

    self.device_instance.cmdBindPipeline(commandBuffer, .graphics, self.graphicsPipeline);

    const viewport: vk.Viewport = .{
        .x = 0.0,
        .y = 0.0,
        .width = @floatFromInt(self.swapChainExtent.width),
        .height = @floatFromInt(self.swapChainExtent.height),
        .min_depth = 0.0,
        .max_depth = 1.0,
    };

    self.device_instance.cmdSetViewport(commandBuffer, 0, 1, @ptrCast(&viewport));

    // var x: i32 = @intCast(self.swapChainExtent.width);
    // var y: i32 = @intCast(self.swapChainExtent.height);
    // x = @divTrunc(x, 2);
    // y = @divTrunc(y, 2);
    // const scissor: vk.Rect2D = .{
    //     .offset = .{ .x = 0, .y = 0 },
    //     .extent = .{ .width = self.swapChainExtent.height / 2, .height = self.swapChainExtent.height / 2 },
    // };
    const scissor: vk.Rect2D = .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.swapChainExtent,
    };

    const s = [_]vk.Rect2D{scissor};

    self.device_instance.cmdSetScissor(commandBuffer, 0, s.len, &s);

    const offsets = [_]vk.DeviceSize{0};
    self.device_instance.cmdBindVertexBuffers(commandBuffer, 0, 1, @ptrCast(&self.vertexBuffer), &offsets);
    self.device_instance.cmdBindIndexBuffer(commandBuffer, self.indexBuffer, 0, .uint32);

    const set = self.descriptorSets.items.ptr + self.currentFrame;
    self.device_instance.cmdBindDescriptorSets(
        commandBuffer,
        .graphics,
        self.pipelineLayout,
        0,
        1,
        set,
        0,
        null,
    );
    self.device_instance.cmdDrawIndexed(
        commandBuffer,
        @intCast(self.mesh.indices.items.len),
        1,
        0,
        0,
        0,
    );

    self.device_instance.cmdEndRenderPass(commandBuffer);

    try self.device_instance.endCommandBuffer(commandBuffer);
}

fn createSyncObjects(self: *Self) !void {
    const semaphoreInfo: vk.SemaphoreCreateInfo = .{};

    const fenceInfo: vk.FenceCreateInfo = .{
        .flags = .{ .signaled_bit = true },
    };

    try self.imageAvailableSemaphores.resize(max_frames_in_flights);
    try self.renderFinishedSemaphores.resize(max_frames_in_flights);
    try self.inFlightFence.resize(max_frames_in_flights);

    for (self.imageAvailableSemaphores.items, self.renderFinishedSemaphores.items, self.inFlightFence.items) |*image, *render, *fence| {
        image.* = try self.device_instance.createSemaphore(&semaphoreInfo, null);
        render.* = try self.device_instance.createSemaphore(&semaphoreInfo, null);
        fence.* = try self.device_instance.createFence(&fenceInfo, null);
    }
}

var start_time: i128 = undefined;

fn init_global() void {
    start_time = std.time.nanoTimestamp();
}

fn rotateZ(angle: f32) zmath.Mat {
    const s = std.math.sin(angle);
    const c = std.math.cos(angle);

    const m00 = c;
    const m01 = s;
    const m10 = -s;
    const m11 = c;

    return .{
        zmath.f32x4(m00, m01, 0.0, 0.0),
        zmath.f32x4(m10, m11, 0.0, 0.0),
        zmath.f32x4(0.0, 0.0, 1.0, 0.0),
        zmath.f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

fn rotateY(angle: f32) zmath.Mat {
    const s = std.math.sin(angle);
    const c = std.math.cos(angle);

    return .{
        zmath.f32x4(c, s, 0.0, 0.0),
        zmath.f32x4(-s, c, 0.0, 0.0),
        zmath.f32x4(0.0, 0.0, 1.0, 0.0),
        zmath.f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

fn updateUniformBuffer(self: Self, currentImage: u32) !void {
    const current_time = std.time.nanoTimestamp();
    const time: f64 = @floatFromInt(current_time - start_time);
    const t: f64 = time / std.time.ns_per_s;
    const angle: f32 = @floatCast(std.math.pi * 0.5 * t);

    // const model: zmath.Mat = rotateZ(angle);

    const width: f32 = @floatFromInt(self.swapChainExtent.width);
    const height: f32 = @floatFromInt(self.swapChainExtent.height);
    const s = width / height;

    const quat = zmath.quatFromAxisAngle(.{ 0.0, 0.0, 1.0, 1.0 }, angle);
    const model = zmath.matFromQuat(quat);
    var ubo: UniformBufferObject = .{
        .model = model,
        .view = zmath.lookAtRh(
            .{ 3.0, 3.0, -3.0, 1.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, 1.0, 0.0 },
        ),
        .proj = zmath.perspectiveFovRh(
            std.math.pi * 0.25,
            s,
            0.1,
            10.0,
        ),
    };

    const dst_p: [*]u8 = @ptrCast(self.uniformBufferMapped.items[currentImage]);
    const src: [*]u8 = @ptrCast(&ubo);
    const size = @sizeOf(UniformBufferObject);
    @memcpy(dst_p[0..size], src[0..size]);
}

fn createDescriptorPool(self: *Self) !void {
    const poolSizes = [_]vk.DescriptorPoolSize{ .{
        .type = .uniform_buffer,
        .descriptor_count = max_frames_in_flights,
    }, .{
        .type = .combined_image_sampler,
        .descriptor_count = max_frames_in_flights,
    } };

    const poolInfo: vk.DescriptorPoolCreateInfo = .{
        .pool_size_count = poolSizes.len,
        .p_pool_sizes = @ptrCast(&poolSizes),
        .max_sets = max_frames_in_flights,
    };

    self.descriptorPool = try self.device_instance.createDescriptorPool(&poolInfo, null);
}

fn createDescriptorSets(self: *Self) !void {
    const layouts = [_]vk.DescriptorSetLayout{ self.descriptorSetLayout, self.descriptorSetLayout };

    const allocInfo: vk.DescriptorSetAllocateInfo = .{
        .descriptor_pool = self.descriptorPool,
        .descriptor_set_count = max_frames_in_flights,
        .p_set_layouts = @ptrCast(&layouts),
    };

    try self.descriptorSets.resize(max_frames_in_flights);

    try self.device_instance.allocateDescriptorSets(&allocInfo, self.descriptorSets.items.ptr);

    for (self.descriptorSets.items, self.uniformBuffers.items) |item, buffer| {
        const bufferInfo: vk.DescriptorBufferInfo = .{
            .buffer = buffer,
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        const imageInfo: vk.DescriptorImageInfo = .{
            .image_layout = .shader_read_only_optimal,
            .image_view = self.texture_image_view,
            .sampler = self.texture_sampler,
        };

        const descriptorWrites = [_]vk.WriteDescriptorSet{
            .{
                .dst_set = item,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .p_buffer_info = @ptrCast(&bufferInfo),
                .p_image_info = @ptrCast(&imageInfo),
                .p_texel_buffer_view = &[_]vk.BufferView{},
            },
            .{
                .dst_set = item,
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = 1,
                .p_image_info = @ptrCast(&imageInfo),
                .p_buffer_info = @ptrCast(&bufferInfo),
                .p_texel_buffer_view = &[_]vk.BufferView{},
            },
        };

        self.device_instance.updateDescriptorSets(descriptorWrites.len, @ptrCast(&descriptorWrites), 0, null);
    }
}

fn drawFrame(self: *Self) !void {
    _ = try self.device_instance.waitForFences(1, self.inFlightFence.items.ptr + self.currentFrame, vk.TRUE, std.math.maxInt(u64));

    const fence = self.inFlightFence.items.ptr + self.currentFrame;
    const commandBuffer = self.commandBuffer.items[self.currentFrame];
    const imageSemaphore = self.imageAvailableSemaphores.items.ptr + self.currentFrame;

    const aq = try self.device_instance.acquireNextImageKHR(
        self.swapChain,
        std.math.maxInt(u64),
        imageSemaphore[0],
        .null_handle,
    );

    switch (aq.result) {
        .error_out_of_date_khr => {
            try self.recreateSwapChain();
            return;
        },
        .success, .suboptimal_khr => {},
        else => @panic("failed to acquire swap chain image!"),
    }

    const imageIndex = aq.image_index;

    const signalSemaphores = self.renderFinishedSemaphores.items.ptr + self.currentFrame;

    try self.updateUniformBuffer(self.currentFrame);
    try self.device_instance.resetFences(1, fence);

    try self.device_instance.resetCommandBuffer(commandBuffer, .fromInt(0));
    try self.recordCommandBuffer(commandBuffer, imageIndex);

    const waitStages: vk.PipelineStageFlags = .{ .color_attachment_output_bit = true };

    var submitInfo: vk.SubmitInfo = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = imageSemaphore,
        .p_wait_dst_stage_mask = @ptrCast(&waitStages),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&commandBuffer),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = signalSemaphores,
    };

    try self.device_instance.queueSubmit(self.graphics_queue, 1, @ptrCast(&submitInfo), fence[0]);

    const presentInfo: vk.PresentInfoKHR = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = signalSemaphores,
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapChain),
        .p_image_indices = @ptrCast(&imageIndex),
    };

    const result = try self.device_instance.queuePresentKHR(self.present_queue, &presentInfo);

    switch (result) {
        .error_out_of_date_khr, .suboptimal_khr => {
            self.framebufferResize = false;
            try self.recreateSwapChain();
        },
        .success => {},
        else => @panic("failed to present swap chain image!"),
    }

    if (self.framebufferResize) {
        self.framebufferResize = false;
        try self.recreateSwapChain();
    }

    self.currentFrame = (self.currentFrame + 1) % max_frames_in_flights;
}

fn recreateSwapChain(self: *Self) !void {
    var width: u32 = 0;
    var height: u32 = 0;
    glfw.getFramebufferSize(self.window, @ptrCast(&width), @ptrCast(&height));

    while (width == 0 or height == 0) {
        glfw.getFramebufferSize(self.window, @ptrCast(&width), @ptrCast(&height));
        glfw.waitEvents();
    }

    try self.device_instance.deviceWaitIdle();
    self.cleanupSwapChain();

    try self.createSwapChain();
    try self.createImageViews();
    try self.createColorResources();
    try self.createDepthResources();
    try self.createFramebuffers();
}

fn cleanupSwapChain(self: Self) void {
    self.device_instance.destroyImageView(self.depth_image_view, null);
    self.device_instance.destroyImage(self.depth_image, null);
    self.device_instance.freeMemory(self.depth_image_memory, null);

    self.device_instance.destroyImageView(self.color_image_view, null);
    self.device_instance.destroyImage(self.color_image, null);
    self.device_instance.freeMemory(self.color_image_memory, null);

    for (self.swapChainFramebuffers.items) |item| {
        self.device_instance.destroyFramebuffer(item, null);
    }

    for (self.swapchainImageViews.items) |view| {
        self.device_instance.destroyImageView(view, null);
    }

    self.device_instance.destroySwapchainKHR(self.swapChain, null);
}
fn createShaderModule(self: Self, code: []align(4) const u8) !vk.ShaderModule {
    const createInfo: vk.ShaderModuleCreateInfo = .{
        .code_size = code.len,
        .p_code = @ptrCast(code.ptr),
    };

    return try self.device_instance.createShaderModule(&createInfo, null);
}

fn chooseSwapSurfaceFormat(availableFormats: []vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    for (availableFormats) |item| {
        if (item.format == .b8g8r8a8_srgb and item.color_space == .srgb_nonlinear_khr) {
            return item;
        }
    }

    return availableFormats[0];
}

fn chooseSwapPresentMode(avaiablePresentModes: []vk.PresentModeKHR) vk.PresentModeKHR {
    for (avaiablePresentModes) |item| {
        if (item == .mailbox_khr) {
            return item;
        }
    }

    return .fifo_khr;
}

fn chooseSwapExtent(self: Self, capabilities: *const vk.SurfaceCapabilitiesKHR) vk.Extent2D {
    if (capabilities.current_extent.width != std.math.maxInt(u32)) {
        return capabilities.current_extent;
    }
    var width: u32 = 0;
    var height: u32 = 0;

    glfw.getFramebufferSize(self.window, @ptrCast(&width), @ptrCast(&height));

    width = std.math.clamp(width, capabilities.min_image_extent.width, capabilities.max_image_extent.width);
    height = std.math.clamp(height, capabilities.min_image_extent.height, capabilities.max_image_extent.height);

    return .{
        .width = width,
        .height = height,
    };
}

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR,
    formats: ?ArrType = null,
    presentModes: ?ArrPreType = null,

    const ArrType = std.ArrayList(vk.SurfaceFormatKHR);
    const ArrPreType = std.ArrayList(vk.PresentModeKHR);

    fn swapChainAdequate(self: SwapChainSupportDetails) bool {
        const result = if (self.formats) |f| f.items.len != 0 else false;
        if (!result) return false;
        return if (self.presentModes) |p| p.items.len != 0 else false;
    }

    fn resize_format(self: *SwapChainSupportDetails, size: u32) !void {
        if (self.formats == null) {
            self.formats = ArrType.init(allocator);
        }
        try self.formats.?.resize(size);
    }

    fn resize_pre(self: *SwapChainSupportDetails, size: u32) !void {
        if (self.presentModes == null) {
            self.presentModes = ArrPreType.init(allocator);
        }
        try self.presentModes.?.resize(size);
    }

    fn deinit(self: *SwapChainSupportDetails) void {
        if (self.formats) |f| {
            f.deinit();
        }
        if (self.presentModes) |pre| {
            pre.deinit();
        }
        self.formats = null;
        self.presentModes = null;
    }
};

fn isDeviceSuitable(self: Self, device: vk.PhysicalDevice) !bool {
    const indices = try self.findQueueFamilies(device);
    print("indices: {any}", .{indices});

    const ext_support = try self.checkDeviceExtensionSupport(device);
    print("support: {any}", .{ext_support});

    var swapChainAdequate = false;
    if (ext_support) {
        var swapChainSupport = try self.querySwapChainSupport(device);
        defer swapChainSupport.deinit();
        swapChainAdequate = swapChainSupport.swapChainAdequate();
    }

    return indices.isComplete() and ext_support and swapChainAdequate;
}

fn querySwapChainSupport(self: Self, device: vk.PhysicalDevice) !SwapChainSupportDetails {
    const cap = try self.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(device, self.surface_enum());
    var details: SwapChainSupportDetails = .{
        .capabilities = cap,
    };

    var format_count: u32 = 0;
    _ = try self.instance.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface_enum(), &format_count, null);

    if (format_count != 0) {
        try details.resize_format(format_count);
        _ = try self.instance.getPhysicalDeviceSurfaceFormatsKHR(device, self.surface_enum(), &format_count, details.formats.?.items.ptr);
    }

    var present_count: u32 = 0;
    _ = try self.instance.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface_enum(), &present_count, null);

    if (present_count != 0) {
        try details.resize_pre(present_count);
        _ = try self.instance.getPhysicalDeviceSurfacePresentModesKHR(device, self.surface_enum(), &present_count, details.presentModes.?.items.ptr);
    }

    return details;
}

/// 检查设配扩展是否支持
fn checkDeviceExtensionSupport(self: Self, device: vk.PhysicalDevice) !bool {
    var ext_count: u32 = 0;
    _ = try self.instance.enumerateDeviceExtensionProperties(device, null, &ext_count, null);

    const arrType = std.ArrayList(vk.ExtensionProperties);
    var arr = arrType.init(allocator);
    defer arr.deinit();

    try arr.resize(ext_count);
    _ = try self.instance.enumerateDeviceExtensionProperties(device, null, &ext_count, arr.items.ptr);

    var device_ext_clone = deviceExtensions;

    var req_exts: [][*:0]const u8 = device_ext_clone[0..];
    for (arr.items) |ext| {
        const name = std.mem.trimRight(u8, ext.extension_name[0..], &[_]u8{0});

        for (req_exts, 0..) |item, index| {
            const len = std.mem.len(item);

            if (std.mem.eql(u8, name, item[0..len])) {
                req_exts = req_exts[(index + 1)..];
                break;
            }
        }
        if (req_exts.len == 0) {
            return true;
        }
    }

    return req_exts.len == 0;
}

const QueueFamilyIndices = struct {
    graphicsFaimily: ?u32 = null,
    presentFamily: ?u32 = null,
    fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphicsFaimily != null and self.presentFamily != null;
    }
};

fn findQueueFamilies(self: Self, device: vk.PhysicalDevice) !QueueFamilyIndices {
    var indice: QueueFamilyIndices = .{};
    var queueFamilyCount: u32 = 0;
    self.instance.getPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

    const arrType = std.ArrayList(vk.QueueFamilyProperties);
    var arr = try arrType.initCapacity(allocator, queueFamilyCount);
    defer arr.deinit();

    try arr.resize(queueFamilyCount);
    self.instance.getPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, arr.items.ptr);

    for (arr.items, 0..) |item, index| {
        const i: u32 = @intCast(index);
        if (item.queue_flags.graphics_bit) {
            indice.graphicsFaimily = i;
        }

        const present_support =
            try self.instance.getPhysicalDeviceSurfaceSupportKHR(device, i, @enumFromInt(self.vk_surface));

        if (present_support == vk.TRUE) {
            indice.presentFamily = i;
        }

        if (indice.isComplete()) {
            break;
        }
    }
    return indice;
}
inline fn print(comptime fmt: []const u8, args: anytype) void {
    if (builtin.mode == .Debug) {
        std.debug.print(fmt ++ "\n", args);
    }
}

fn readFile(file_name: []const u8) ![]align(4) const u8 {
    const file = try std.fs.cwd().openFile(file_name, .{});
    defer file.close();

    const size = (try file.metadata()).size();
    const bytes = try file.readToEndAllocOptions(
        allocator,
        size,
        null,
        4,
        null,
    );

    return bytes;
}

const UniformBufferObject = struct {
    model: zmath.Mat align(16),
    view: zmath.Mat align(16),
    proj: zmath.Mat align(16),
};

pub fn beginSingleTimeCommands(self: *Self) !vk.CommandBuffer {
    const alloc_info = vk.CommandBufferAllocateInfo{
        .command_pool = self.commandPool,
        .level = .primary,
        .command_buffer_count = 1,
    };

    var command_buffer: vk.CommandBuffer = undefined;
    try self.device_instance.allocateCommandBuffers(&alloc_info, @ptrCast(&command_buffer));

    const begin_info = vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    };

    try self.device_instance.beginCommandBuffer(command_buffer, &begin_info);
    return command_buffer;
}

pub fn endSingleTimeCommands(self: *Self, command_buffer: vk.CommandBuffer) !void {
    try self.device_instance.endCommandBuffer(command_buffer);

    const submit_info = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = null,
        .p_wait_dst_stage_mask = null,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&command_buffer),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = null,
    };

    try self.device_instance.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit_info), .null_handle);
    try self.device_instance.queueWaitIdle(self.graphics_queue);
    self.device_instance.freeCommandBuffers(self.commandPool, 1, @ptrCast(&command_buffer));
}
