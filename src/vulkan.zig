const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("zglfw");
const vk = @import("vulkan");
const zlm = @import("zlm");
const zmath = @import("zmath");

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const allocator = gpa.allocator();

const Self = @This();

window: *glfw.Window,
vkb: vk.BaseWrapper,
instance: vk.InstanceProxy = undefined,
// hide
_vk_inst_wrapper: vk.InstanceWrapper = undefined,

debugMessage: ?vk.DebugUtilsMessengerEXT = null,
vk_surface: glfw.VkSurfaceKHR = undefined,

// 物理设备
vk_physical_device: ?vk.PhysicalDevice = null,
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
    return .{ .window = window, .vkb = vkb };
}

fn framebuferResizeCallback(window: ?*glfw.Window, width: c_int, height: c_int) callconv(.c) void {
    _ = width;
    _ = height;
    const v: usize = @intFromPtr(window.?);
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

const enableValidationLayers = builtin.mode == .Debug;

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
    print("done...", .{});
    try self.createGraphicsPipeline();

    try self.createFramebuffers();

    try self.createCommandPool();

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

    var createInfo: vk.InstanceCreateInfo = .{ .p_application_info = &app_info };
    var count: u32 = undefined;
    const exts = glfw.getRequiredInstanceExtensions(&count) orelse &[0][*:0]const u8{};
    const arrType = std.ArrayList([*:0]const u8);
    var arr = try arrType.initCapacity(allocator, count);
    defer arr.deinit();

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
            self.vk_physical_device = item;
            print("physical device: {any}", .{item});
            break;
        }
    }

    if (self.vk_physical_device) |_| {} else {
        @panic("failed to find a suitable GPU!");
    }
}

fn createLogicalDevice(self: *Self) !void {
    const indices = try self.findQueueFamilies(self.vk_physical_device.?);

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
        self.instance.getPhysicalDeviceFeatures(self.vk_physical_device.?);

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

    const vk_device = try self.instance.createDevice(self.vk_physical_device.?, &createInfo, null);

    self._device_wrapper = vk.DeviceWrapper.load(vk_device, self.instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
    self.device_instance = vk.DeviceProxy.init(vk_device, &self._device_wrapper);

    self.graphics_queue = self.device_instance.getDeviceQueue(indices.graphicsFaimily.?, 0);
    self.present_queue = self.device_instance.getDeviceQueue(indices.presentFamily.?, 0);
}

fn createSwapChain(self: *Self) !void {
    var swap_chain_support = try self.querySwapChainSupport(self.vk_physical_device.?);
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

    const indices = try self.findQueueFamilies(self.vk_physical_device.?);

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
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const colorAttchmentRef: vk.AttachmentReference = .{
        .attachment = 0,
        .layout = .attachment_optimal,
    };

    const subpass: vk.SubpassDescription = .{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&colorAttchmentRef),
    };

    const dependency: vk.SubpassDependency = .{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .fromInt(0),
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    };

    const renderPassInfo: vk.RenderPassCreateInfo = .{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&colorAttachment),
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

    const layoutInfo: vk.DescriptorSetLayoutCreateInfo = .{
        .binding_count = 1,
        .p_bindings = @ptrCast(&uboLayoutBinding),
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

    const bindingDesc = Vertex.getBindingDescription();
    const attributeDesc = Vertex.getAttributeDescriptions();
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
        .rasterization_samples = .{ .@"1_bit" = true },
        .alpha_to_coverage_enable = 0,
        .alpha_to_one_enable = 0,
        .min_sample_shading = 0,
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

    for (self.swapchainImageViews.items, self.swapChainFramebuffers.items) |*item, *buf| {
        const framebufferInfo: vk.FramebufferCreateInfo = .{
            .render_pass = self.renderPass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(item),
            .width = self.swapChainExtent.width,
            .height = self.swapChainExtent.height,
            .layers = 1,
        };

        buf.* = try self.device_instance.createFramebuffer(&framebufferInfo, null);
    }
}

fn createCommandPool(self: *Self) !void {
    const queueFamilyIndices = try self.findQueueFamilies(self.vk_physical_device.?);
    const poolInfo: vk.CommandPoolCreateInfo = .{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queueFamilyIndices.graphicsFaimily.?,
    };

    self.commandPool = try self.device_instance.createCommandPool(&poolInfo, null);
}

fn createVertexBuffer(self: *Self) !void {
    const bufferSize: vk.DeviceSize = comptime @sizeOf(@TypeOf(vertices[0])) * vertices.len;

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
    const src: []const u8 = @ptrCast(&vertices);
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
    const bufferSize: vk.DeviceSize = comptime @sizeOf(@TypeOf(indices_g[0])) * indices_g.len;

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
    const src: []const u8 = @ptrCast(&indices_g);
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
    const memProperties = self.instance.getPhysicalDeviceMemoryProperties(self.vk_physical_device.?);

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

    const clearColor: vk.ClearColorValue = .{
        .float_32 = .{ 0.0, 0.0, 0.0, 0.0 },
    };

    renderPassInfo.clear_value_count = 1;
    renderPassInfo.p_clear_values = @ptrCast(&clearColor);

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
    self.device_instance.cmdBindIndexBuffer(commandBuffer, self.indexBuffer, 0, .uint16);

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
        @intCast(indices_g.len),
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

fn updateUniformBuffer(self: Self, currentImage: u32) !void {
    const current_time = std.time.nanoTimestamp();
    const time: f64 = @floatFromInt(current_time - start_time);
    const t: f64 = time / std.time.ns_per_s;
    const angle: f32 = @floatCast(std.math.pi * 0.5 * t);

    const model: zmath.Mat = zmath.rotationZ(angle);

    const width: f32 = @floatFromInt(self.swapChainExtent.width);
    const height: f32 = @floatFromInt(self.swapChainExtent.height);
    const s = width / height;

    var ubo: UniformBufferObject = .{
        .model = model,
        .view = zmath.lookAtRh(
            .{ 2.0, 2.0, 2.0, 1.0 },
            .{ 0.0, 0.0, 0.0, 0.0 },
            .{ 0.0, 0.0, -1.0, 0.0 },
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
    const poolSize: vk.DescriptorPoolSize = .{
        .descriptor_count = max_frames_in_flights,
        .type = .uniform_buffer,
    };

    const poolInfo: vk.DescriptorPoolCreateInfo = .{
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&poolSize),
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

        const descriptorWrite: vk.WriteDescriptorSet = .{
            .dst_set = item,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .p_buffer_info = @ptrCast(&bufferInfo),
            .p_image_info = @ptrCast(&[_]vk.DescriptorBufferInfo{}),
            .p_texel_buffer_view = @ptrCast(&[_]vk.BufferView{}),
        };

        self.device_instance.updateDescriptorSets(1, @ptrCast(&descriptorWrite), 0, null);
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
            if (self.framebufferResize) {
                self.framebufferResize = false;
                try self.recreateSwapChain();
            }
        },
        .success => {},
        else => @panic("failed to present swap chain image!"),
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
    try self.createFramebuffers();
}

fn cleanupSwapChain(self: Self) void {
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

const Vertex = struct {
    pos: zmath.Vec,
    color: zmath.Vec,

    fn getBindingDescription() vk.VertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .input_rate = .vertex,
        };
    }

    fn getAttributeDescriptions() []const vk.VertexInputAttributeDescription {
        return &.{
            .{
                .binding = 0,
                .location = 0,
                .format = .r32g32b32a32_sfloat,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = .r32g32b32a32_sfloat,
                .offset = @offsetOf(Vertex, "color"),
            },
        };
    }
};

const vertices = [_]Vertex{
    .{ .pos = .{ -0.5, -0.5, 0.0, 1.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } },
    .{ .pos = .{ 0.5, -0.5, 0.0, 1.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } },
    .{ .pos = .{ 0.5, 0.5, 0.0, 1.0 }, .color = .{ 0.0, 0.0, 1.0, 1.0 } },
    .{ .pos = .{ -0.5, 0.5, 0.0, 1.0 }, .color = .{ 1.0, 1.0, 1.0, 1.0 } },
};

const indices_g = [_]u16{ 0, 1, 2, 2, 3, 0 };
