#include "selftest.h"
#include "../../runtime/d3d12/compute_fixture.h"
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>
#include "xbox_shader.h"

namespace {
using compute_fixture::buffer;
using frame_abi::check;
using frame_abi::ComPtr;
using frame_abi::method;

// The Scarlett release record with ID3D12Device11 selects the queue and
// command-list tables a Series X|S title built with a recent GDK uses.
constexpr std::array<UINT, 18> scarlett_profile{
    0x20004, 0, UINT_MAX, 0, 0, 0, UINT_MAX, UINT_MAX, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};

// A queue held on a label is released by the provider after five seconds, so a
// scenario that needs that release must already count as failed.
constexpr DWORD deadline_ms = 2000;

// Xbox command-list and queue slots (GDK gen9/d3d12_xs.h).
constexpr size_t slot_copy_memory = 62, slot_write32 = 71, slot_write64 = 72, slot_wait64 = 74,
                 slot_copy_texture = 83, slot_build = 119, slot_sync_top = 131, slot_sync_bottom = 132;
constexpr size_t slot_compute_root = 33, device_create_compute = 62;
constexpr size_t queue_signal = 16, queue_execute_frame = 35, device_place = 44, device_create_list = 73;

using CopyMemory_ = void (WINAPI*)(ID3D12GraphicsCommandList*, UINT64, UINT64, UINT64, UINT);
using Write32 = void (WINAPI*)(ID3D12GraphicsCommandList*, UINT64, UINT, UINT, UINT);
using Write64 = void (WINAPI*)(ID3D12GraphicsCommandList*, UINT64, UINT64, UINT, UINT);
using Wait64 = void (WINAPI*)(ID3D12GraphicsCommandList*, UINT64, D3D12_COMPARISON_FUNC, UINT64, UINT);
using Build = void (WINAPI*)(ID3D12GraphicsCommandList*, const void*, UINT, const void*);
using Sync = void (WINAPI*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT64, UINT64);
using SignalQueue = HRESULT (WINAPI*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
using ExecuteFrame = void (WINAPI*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*, UINT64, UINT);
using CreateComputePipeline = HRESULT (WINAPI*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*,
                                                UINT, const void*, REFIID, void**);
using SetRoot = void (WINAPI*)(ID3D12GraphicsCommandList*, ID3D12RootSignature*);
using Place = HRESULT (WINAPI*)(ID3D12Device*, UINT64, const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                const D3D12_CLEAR_VALUE*, REFIID, void**);

// Committed title memory. A Series title addresses the GPU with these pointers.
struct Memory {
    BYTE* base;
    explicit Memory(size_t bytes)
        : base(static_cast<BYTE*>(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE))) {
        if (!base) throw std::runtime_error("allocate title memory");
    }
    ~Memory() { VirtualFree(base, 0, MEM_RELEASE); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    UINT64 address(size_t offset = 0) const { return reinterpret_cast<UINT64>(base + offset); }
    UINT64 read64(size_t offset) const { return *reinterpret_cast<volatile const UINT64*>(base + offset); }
};

struct Queue {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 submitted = 0;

    Queue(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = type;
        check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "create queue");
        check(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)), "create allocator");
        const UINT record[] = {UINT(type), 0, 1, 0};
        check(method<frame_abi::CreateList>(device, device_create_list)(
                  device, record, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "create list");
        check(list->Reset(allocator.Get(), nullptr), "open list");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create fence");
    }
    void write32(UINT64 address, UINT value, UINT flush = 0) {
        method<Write32>(list.Get(), slot_write32)(list.Get(), address, value, flush, 0);
    }
    void write64(UINT64 address, UINT64 value, UINT flush = 0) {
        method<Write64>(list.Get(), slot_write64)(list.Get(), address, value, flush, 0);
    }
    void wait64(UINT64 address, UINT64 value) {
        method<Wait64>(list.Get(), slot_wait64)(list.Get(), address, D3D12_COMPARISON_FUNC_GREATER_EQUAL, value, 0);
    }
    void close() { check(list->Close(), "close list"); }
    void signal() {
        check(method<SignalQueue>(queue.Get(), queue_signal)(queue.Get(), fence.Get(), ++submitted), "signal queue");
    }
    void submit() {
        close();
        ID3D12CommandList* batch[]{list.Get()};
        queue->ExecuteCommandLists(1, batch);
        signal();
    }
    // Wait until everything submitted has finished, or fail.
    void finish(const char* what) {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("create completion event");
        const HRESULT armed = fence->SetEventOnCompletion(submitted, event);
        const DWORD waited = SUCCEEDED(armed) ? WaitForSingleObject(event, deadline_ms) : WAIT_FAILED;
        CloseHandle(event);
        check(armed, "arm completion event");
        if (waited != WAIT_OBJECT_0)
            throw std::runtime_error(std::string(what) + " did not finish within 2 seconds");
    }
};

void expect_value(const Memory& memory, size_t offset, UINT64 value, const char* what) {
    const UINT64 found = memory.read64(offset);
    if (found != value) {
        char message[160];
        std::snprintf(message, sizeof(message), "%s: expected %llu, found %llu", what,
                      static_cast<unsigned long long>(value), static_cast<unsigned long long>(found));
        throw std::runtime_error(message);
    }
}

// Both bottom-of-pipe writes store ordinary values, with or without a flush.
void labels(ID3D12Device* device) {
    Memory memory(65536);
    Queue direct(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    direct.write32(memory.address(0), 0x12345678);
    direct.write32(memory.address(8), 7, 1);
    direct.write64(memory.address(16), 0x1122334455667788ull);
    direct.write64(memory.address(24), 42, 0x100);
    direct.submit();
    direct.finish("label writes");
    expect_value(memory, 0, 0x12345678, "32-bit write");
    expect_value(memory, 8, 7, "32-bit write with a flush");
    expect_value(memory, 16, 0x1122334455667788ull, "64-bit write");
    expect_value(memory, 24, 42, "64-bit write with a flush");
}

// A compute queue waits for a label the direct queue writes later, and the
// direct queue then waits for the compute queue's answer.
void cross_queue(ID3D12Device* device) {
    Memory memory(65536);
    Queue compute(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    Queue direct(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    compute.wait64(memory.address(0), 5);
    compute.write64(memory.address(8), 6);
    compute.submit();
    direct.write64(memory.address(0), 5);
    direct.submit();
    direct.finish("direct queue");
    compute.finish("compute queue waiting on a direct queue label");
    expect_value(memory, 8, 6, "compute answer");
}

// The compute queue submits through ExecuteCommandListsX as well.
void compute_frame(ID3D12Device* device) {
    Memory memory(65536);
    Queue compute(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    compute.write64(memory.address(0), 99);
    compute.close();
    ID3D12CommandList* batch[]{compute.list.Get()};
    method<ExecuteFrame>(compute.queue.Get(), queue_execute_frame)(compute.queue.Get(), 1, batch, 1, 3);
    compute.signal();
    compute.finish("compute frame submission");
    expect_value(memory, 0, 99, "compute frame write");
}

// A command that cannot be recorded is left out on its own. The list still
// closes and its fence write, which another queue may need, still happens.
void skipped_command(ID3D12Device* device) {
    Memory memory(65536);
    Queue direct(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto sync_top = method<Sync>(direct.list.Get(), slot_sync_top);
    auto sync_bottom = method<Sync>(direct.list.Get(), slot_sync_bottom);
    sync_top(direct.list.Get(), 0xc280, 0x20000, 0, 0xffffffffff00ull);
    // The BVH builder passes no description for a build it cannot express.
    method<Build>(direct.list.Get(), slot_build)(direct.list.Get(), nullptr, 0, nullptr);
    sync_bottom(direct.list.Get(), 0xc280, 0x20000, 0, 0xffffffffff00ull);
    direct.write64(memory.address(0), 11);
    direct.submit();
    direct.finish("list with a skipped command");
    expect_value(memory, 0, 11, "write after a skipped command");
}

// CopyMemoryToMemoryX copies from the source to the destination.
void copy_memory(ID3D12Device* device) {
    constexpr size_t bytes = 4096, source = 0, destination = 65536;
    Memory memory(2 * 65536);
    for (size_t i = 0; i < bytes; ++i) {
        memory.base[source + i] = static_cast<BYTE>(i * 31 + 7);
        memory.base[destination + i] = 0xee;
    }
    Queue direct(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    method<CopyMemory_>(direct.list.Get(), slot_copy_memory)(
        direct.list.Get(), memory.address(destination), memory.address(source), bytes, 0);
    direct.submit();
    direct.finish("memory copy");
    if (std::memcmp(memory.base + source, memory.base + destination, bytes))
        throw std::runtime_error("memory copy did not reach the destination");
    if (memory.base[source + 1] != 38)
        throw std::runtime_error("memory copy overwrote its source");
}

// The compute queue waits for a label, then reads a guest texture the direct
// queue writes before setting that label. A provider that makes the writer wait
// for the earlier submitted reader deadlocks the two queues.
void shared_texture(ID3D12Device* device) {
    constexpr UINT width = 256, height = 256, pitch = 256;
    constexpr UINT64 bytes = UINT64(pitch) * height;
    Memory memory(4 * 65536);
    Memory label(65536);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R8_UNORM;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x105);
    ComPtr<ID3D12Resource> texture;
    check(method<Place>(device, device_place)(device, memory.address(), &desc, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(&texture)), "place guest texture");
    auto upload = buffer(device, D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(device, D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST);
    std::vector<BYTE> expected(bytes);
    for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<BYTE>(i * 13 + i / pitch);
    compute_fixture::upload(upload.Get(), expected.data(), bytes);

    D3D12_TEXTURE_COPY_LOCATION image{}, linear{};
    image.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    image.pResource = texture.Get();
    linear.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    linear.PlacedFootprint.Footprint = {DXGI_FORMAT_R8_UNORM, width, height, 1, pitch};

    Queue compute(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    Queue direct(device, D3D12_COMMAND_LIST_TYPE_DIRECT);
    compute.wait64(label.address(), 1);
    linear.pResource = readback.Get();
    method<frame_abi::Copy>(compute.list.Get(), slot_copy_texture)(
        compute.list.Get(), &linear, 0, 0, 0, &image, nullptr, 0);
    compute.submit();
    linear.pResource = upload.Get();
    method<frame_abi::Copy>(direct.list.Get(), slot_copy_texture)(
        direct.list.Get(), &image, 0, 0, 0, &linear, nullptr, 0);
    direct.write64(label.address(), 1);
    direct.submit();
    direct.finish("direct queue writing a texture a waiting queue reads");
    compute.finish("compute queue reading the texture");

    D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)}, empty{};
    void* mapped{};
    check(readback->Map(0, &range, &mapped), "map readback");
    const bool matched = std::memcmp(mapped, expected.data(), bytes) == 0;
    readback->Unmap(0, &empty);
    if (!matched) throw std::runtime_error("the reader did not see the writer's texels");
}

// A console shader reaches the GPU through the translation cache. The provider
// substitutes an empty pipeline when no translation exists, so `verify` also
// checks that it reports loading the translated one.
void xbox_shader_scenario(ID3D12Device* device) {
    ComPtr<ID3D12RootSignature> root;
    D3D12_ROOT_SIGNATURE_DESC signature{};
    ComPtr<ID3DBlob> serialized;
    check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, nullptr),
          "serialize empty root signature");
    check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "create empty root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC shader{};
    shader.pRootSignature = root.Get();
    shader.CS = {xbox_shader, sizeof(xbox_shader)};
    ComPtr<ID3D12PipelineState> pipeline;
    check(method<CreateComputePipeline>(device, device_create_compute)(device, &shader, 0, nullptr,
                                                                       IID_PPV_ARGS(&pipeline)),
          "create pipeline from a console shader");
    Memory memory(65536);
    Queue compute(device, D3D12_COMMAND_LIST_TYPE_COMPUTE);
    method<SetRoot>(compute.list.Get(), slot_compute_root)(compute.list.Get(), root.Get());
    compute.list->SetPipelineState(pipeline.Get());
    compute.list->Dispatch(1, 1, 1);
    compute.write64(memory.address(0), 3);
    compute.submit();
    compute.finish("dispatch of the console shader");
    expect_value(memory, 0, 3, "write after the dispatch");
}

struct Scenario {
    const wchar_t* name;
    void (*run)(ID3D12Device*);
};
constexpr Scenario scenarios[] = {
    {L"labels", labels},
    {L"cross-queue", cross_queue},
    {L"compute-frame", compute_frame},
    {L"skipped-command", skipped_command},
    {L"copy-memory", copy_memory},
    {L"shared-texture", shared_texture},
    {L"xbox-shader", xbox_shader_scenario},
};

// A fresh Scarlett device on the requested adapter. The factory is reached
// through the Xbox device, so the game imports no desktop graphics library.
ComPtr<ID3D12Device> create_scarlett_device(const wchar_t* requested, char (&pci)[16]) {
    auto create = [](IUnknown* adapter, ComPtr<ID3D12Device>& out) {
        return D3D12XboxCreateDevice(adapter, scarlett_profile.data(), __uuidof(ID3D12Device11),
                                     reinterpret_cast<void**>(out.ReleaseAndGetAddressOf()));
    };
    auto matches = [&](IDXGIAdapter1* adapter) {
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc), "adapter description");
        std::snprintf(pci, sizeof(pci), "%04x:%04x", desc.VendorId, desc.DeviceId);
        wchar_t found[16];
        std::swprintf(found, 16, L"%04x:%04x", desc.VendorId, desc.DeviceId);
        return !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && (!requested || !_wcsicmp(requested, found));
    };
    ComPtr<ID3D12Device> device;
    check(create(nullptr, device), "Scarlett device");
    ComPtr<IDXGIDevice1> dxgi;
    check(device.As(&dxgi), "Xbox device DXGI interface");
    ComPtr<IDXGIAdapter> adapter;
    check(dxgi->GetAdapter(&adapter), "Xbox device adapter");
    ComPtr<IDXGIAdapter1> adapter1;
    check(adapter.As(&adapter1), "Xbox device adapter interface");
    if (matches(adapter1.Get())) return device;
    ComPtr<IDXGIFactory1> factory;
    check(adapter->GetParent(IID_PPV_ARGS(&factory)), "Xbox adapter parent");
    for (UINT n = 0;; ++n) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT hr = factory->EnumAdapters1(n, &candidate);
        if (hr == DXGI_ERROR_NOT_FOUND) throw std::runtime_error("no matching Xbox D3D12 hardware adapter");
        check(hr, "adapter enumeration");
        if (!matches(candidate.Get())) continue;
        check(create(candidate.Get(), device), "Scarlett device on selected adapter");
        return device;
    }
}

std::string narrow(const wchar_t* text) {
    std::string result;
    while (*text) result += static_cast<char>(*text++);
    return result;
}
}

const wchar_t* const selftest_scenarios[] = {L"labels",          L"cross-queue", L"compute-frame",
                                             L"skipped-command", L"copy-memory", L"shared-texture",
                                             L"xbox-shader"};
const unsigned selftest_scenario_count = static_cast<unsigned>(std::size(selftest_scenarios));
static_assert(std::size(selftest_scenarios) == std::size(scenarios));

std::string run_selftest(const wchar_t* requested, const wchar_t* adapter) {
    const bool all = !std::wcscmp(requested, L"all");
    unsigned passed = 0;
    char pci[16]{};
    for (const auto& scenario : scenarios) {
        if (!all && std::wcscmp(requested, scenario.name)) continue;
        const auto name = narrow(scenario.name);
        try {
            auto device = create_scarlett_device(adapter, pci);
            scenario.run(device.Get());
        } catch (const std::exception& error) {
            throw std::runtime_error(name + ": " + error.what());
        }
        ++passed;
        // Progress, so a crash or a stuck later scenario can be located.
        std::fprintf(stderr, "lanes-selftest %s ok\n", name.c_str());
    }
    if (!passed) throw std::runtime_error("unknown scenario " + narrow(requested));
    return "lanes-selftest PASS pci=" + std::string(pci) + " scenario=" + narrow(requested) +
           " passed=" + std::to_string(passed);
}
