// Emulates AMD Zen2-only instructions on Intel CPUs (GTA VI / Xbox Series code).
// VEH: on EXCEPTION_ILLEGAL_INSTRUCTION decode the faulting instruction, do its job, step over.
#include <windows.h>
#include <stdint.h>

static LONG hits[16];
static volatile LONG inited;

static int is_clwb(const uint8_t* c, int* len) {
    // 66 [REX] 0F AE /6 (memory operand)
    int i = 0;
    if (c[0] != 0x66) return 0;
    i = 1;
    int rex = 0;
    if ((c[i] & 0xF0) == 0x40) { rex = c[i]; i++; }
    if (c[i] != 0x0F || c[i+1] != 0xAE) return 0;
    uint8_t modrm = c[i+2];
    if ((modrm >> 6) == 3 || (((modrm >> 3) & 7) != 6 && ((modrm >> 3) & 7) != 7)) return 0;
    int n = i + 3, base = modrm & 7;
    if (base == 4) { base = c[n] & 7; n++; }
    int mode = modrm >> 6;
    if (mode == 0) { if ((modrm & 7) == 5 || ((modrm & 7) == 4 && base == 5)) n += 4; }
    else if (mode == 1) n += 1;
    else n += 4;
    *len = n;
    return 1;
}


// SSE4a on x86 (AMD-only; Intel has no SSE4a). Per APM Vol 3/4 the mandatory
// prefix selects the operation and the opcode selects the form:
//   66 0F 78 /0 ib ib = EXTRQ xmm1(rm), length=imm1[5:0], index=imm2[5:0]
//   66 0F 79 /r       = EXTRQ xmm1(reg), xmm2(rm); length=xmm2[5:0], index=xmm2[13:8]
//   F2 0F 78 /r ib ib = INSERTQ xmm1(reg), xmm2(rm), length=imm1[5:0], index=imm2[5:0]
//   F2 0F 79 /r       = INSERTQ xmm1(reg), xmm2(rm); length=xmm2[69:64], index=xmm2[77:72]
// A length of zero means 64 bits. EXTRQ zero-extends the extracted field into
// the low qword. INSERTQ writes only the low qword of the destination; the
// upper qword is architecturally undefined and is left untouched here.
static int emulate_sse4a(const uint8_t* c, CONTEXT* ctx) {
    int i = 0, has_f2 = 0, has_66 = 0;
    for (;;) {
        uint8_t b = c[i];
        if (b == 0x66) { has_66 = 1; i++; continue; }
        if (b == 0xF2) { has_f2 = 1; i++; continue; }
        if (b == 0xF3 || b == 0x2E || b == 0x3E || b == 0x26 || b == 0x36 || b == 0x64 || b == 0x65 || b == 0x67) { i++; continue; }
        break;
    }
    int rex = 0;
    if ((c[i] & 0xF0) == 0x40) { rex = c[i]; i++; }
    if (c[i] != 0x0F || (c[i + 1] != 0x78 && c[i + 1] != 0x79)) return 0;
    uint8_t op = c[i + 1];
    uint8_t modrm = c[i + 2];
    if ((modrm >> 6) != 3) return 0;
    int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
    int rm = (modrm & 7) | ((rex & 1) ? 8 : 0);
    M128A* xmm = &ctx->Xmm0;
    int consumed = 0;
    if (op == 0x78 && has_f2 && !has_66) {
        // INSERTQ xmm1, xmm2, length, index
        uint32_t length = c[i + 3] & 0x3Fu, index = c[i + 4] & 0x3Fu;
        uint64_t src = (uint64_t)xmm[rm].Low;
        uint64_t dst = (uint64_t)xmm[reg].Low;
        if (length != 0 && length <= 64 && index < 64 && index + length <= 64) {
            uint64_t mask = length == 64 ? ~(uint64_t)0 : (((uint64_t)1 << length) - 1);
            mask <<= index;
            dst = (dst & ~mask) | ((src << index) & mask);
        }
        xmm[reg].Low = dst;
        consumed = i + 5;
    } else if (op == 0x79 && has_f2 && !has_66) {
        // INSERTQ xmm1, xmm2 with the length/index in xmm2[69:64]/xmm2[77:72]
        uint32_t spec = (uint32_t)xmm[rm].High;
        uint32_t length = spec & 0x3Fu, index = (spec >> 8) & 0x3Fu;
        uint64_t src = (uint64_t)xmm[rm].Low;
        uint64_t dst = (uint64_t)xmm[reg].Low;
        if (length != 0 && length <= 64 && index < 64 && index + length <= 64) {
            uint64_t mask = length == 64 ? ~(uint64_t)0 : (((uint64_t)1 << length) - 1);
            mask <<= index;
            dst = (dst & ~mask) | ((src << index) & mask);
        }
        xmm[reg].Low = dst;
        consumed = i + 3;
    } else if (op == 0x78 && has_66 && !has_f2 && ((modrm >> 3) & 7) == 0) {
        // EXTRQ xmm1, length, index (destination is the rm field, /0)
        uint32_t length = c[i + 3] & 0x3Fu, index = c[i + 4] & 0x3Fu;
        uint64_t src = (uint64_t)xmm[rm].Low;
        uint64_t value = 0;
        if (length == 0 && index == 0) value = src;
        else if (length != 0 && length <= 64 && index < 64 && index + length <= 64)
            value = (src >> index) & ((length == 64) ? ~(uint64_t)0 : (((uint64_t)1 << length) - 1));
        xmm[rm].Low = value;
        xmm[rm].High = 0;
        consumed = i + 5;
    } else if (op == 0x79 && has_66 && !has_f2) {
        // EXTRQ xmm1(reg), xmm2(rm); length/index from xmm2 low qword
        uint32_t spec = (uint32_t)xmm[rm].Low;
        uint32_t length = spec & 0x3Fu, index = (spec >> 8) & 0x3Fu;
        uint64_t src = (uint64_t)xmm[reg].Low;
        uint64_t value = 0;
        if (length == 0 && index == 0) value = src;
        else if (length != 0 && length <= 64 && index < 64 && index + length <= 64)
            value = (src >> index) & ((length == 64) ? ~(uint64_t)0 : (((uint64_t)1 << length) - 1));
        xmm[reg].Low = value;
        xmm[reg].High = 0;
        consumed = i + 3;
    } else {
        return 0;
    }
    {
        static int shown;
        if (!shown) {
            shown = 1;
            char b[128];
            wsprintfA(b, "[veh] sse4a %s len=%d\n",
                (has_f2 ? "insertq" : "extrq"), consumed - i);
            OutputDebugStringA(b);
        }
    }
    ctx->Rip += consumed;
    return consumed;
}

static void log_type(int which, const char* name, int note) {
    static int logged[8];
    if (logged[which]) return;
    logged[which] = 1;
    char b[96];
    int n = wsprintfA(b, "[veh] first %s note=%d", name, note);
    char out[112]; int k = 0;
    for (int i = 0; i < n; ++i) out[k++] = b[i];
    out[k++] = '\n'; out[k] = 0;
    OutputDebugStringA(out);
}

// The console instruction is only a hint on a coherent PC, so replace it with
// NOPs the first time it faults. Otherwise a write-back loop faults once per
// cache line and the emulation costs more than the loop itself.
static void patch_nops(void* address, int length) {
    DWORD old = 0;
    if (VirtualProtect(address, (SIZE_T)length, PAGE_EXECUTE_READWRITE, &old)) {
        memset(address, 0x90, (size_t)length);
        DWORD ignored = 0;
        VirtualProtect(address, (SIZE_T)length, old, &ignored);
    }
}

// The title's zeroing loop: `clzero` then a step, an unsigned bound check and a
// short branch back. One fault per cache line costs a user/kernel round trip, so
// emulate the whole range at once and leave rax, the flags and rip as the loop does.
static unsigned long long* gpr(CONTEXT* ctx, int index) {
    static const int order[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    (void)order;
    switch (index) {
    case 0: return (unsigned long long*)&ctx->Rax;
    case 1: return (unsigned long long*)&ctx->Rcx;
    case 2: return (unsigned long long*)&ctx->Rdx;
    case 3: return (unsigned long long*)&ctx->Rbx;
    case 4: return (unsigned long long*)&ctx->Rsp;
    case 5: return (unsigned long long*)&ctx->Rbp;
    case 6: return (unsigned long long*)&ctx->Rsi;
    case 7: return (unsigned long long*)&ctx->Rdi;
    case 8: return (unsigned long long*)&ctx->R8;
    case 9: return (unsigned long long*)&ctx->R9;
    case 10: return (unsigned long long*)&ctx->R10;
    case 11: return (unsigned long long*)&ctx->R11;
    case 12: return (unsigned long long*)&ctx->R12;
    case 13: return (unsigned long long*)&ctx->R13;
    case 14: return (unsigned long long*)&ctx->R14;
    default: return (unsigned long long*)&ctx->R15;
    }
}

static int getenv_bool(const char* name) {
    char value[8];
    DWORD n = GetEnvironmentVariableA(name, value, sizeof value);
    return n > 0 && n < sizeof(value) && (value[0] == '1' || value[0] == 'y' || value[0] == 'Y');
}

static int emulate_clzero_loop(const unsigned char* c, CONTEXT* ctx) {
    if (!(c[0] == 0x0f && c[1] == 0x01 && c[2] == 0xfc)) return 0;
    int i = 3;
    unsigned long long step = 0;
    if (c[i] == 0x48 && c[i + 1] == 0x83 && c[i + 2] == 0xc0) { step = c[i + 3]; i += 4; }
    else if (c[i] == 0x48 && c[i + 1] == 0x81 && c[i + 2] == 0xc0) {
        step = (unsigned long long)c[i + 3] | ((unsigned long long)c[i + 4] << 8) | ((unsigned long long)c[i + 5] << 16) | ((unsigned long long)c[i + 6] << 24);
        i += 7;
    } else if (c[i] == 0x48 && c[i + 1] == 0x05) {
        step = (unsigned long long)c[i + 2] | ((unsigned long long)c[i + 3] << 8) | ((unsigned long long)c[i + 4] << 16) | ((unsigned long long)c[i + 5] << 24);
        i += 6;
    } else return 0;
    if (step == 0) return 0;
    unsigned char rex = c[i];
    if ((rex & 0xf8) != 0x48 || (c[i + 1] != 0x39 && c[i + 1] != 0x3b) || (c[i + 2] >> 6) != 3) return 0;
    int rm = (c[i + 2] & 7) | ((rex & 1) ? 8 : 0);
    int reg = ((c[i + 2] >> 3) & 7) | ((rex & 4) ? 8 : 0);
    int op = c[i + 1];
    i += 3;
    int jb_len = 0;
    long long rel = 0;
    if (c[i] == 0x72) { jb_len = 2; rel = (signed char)c[i + 1]; }
    else if (c[i] == 0x0f && c[i + 1] == 0x82) {
        jb_len = 6;
        rel = (int)c[i + 2] | ((int)c[i + 3] << 8) | ((int)c[i + 4] << 16) | ((int)c[i + 5] << 24);
    } else return 0;
    int loop = i + jb_len;
    if ((uintptr_t)(ctx->Rip + loop + rel) != (uintptr_t)ctx->Rip) return 0;
    int end_index;
    if (op == 0x39) { if (rm != 0) return 0; end_index = reg; }
    else { if (reg != 0) return 0; end_index = rm; }
    unsigned long long start = ctx->Rax;
    unsigned long long limit = *gpr(ctx, end_index);
    if (limit < start || limit - start > (1ull << 28)) return 0;
    static volatile LONG seen_steps;
    if (InterlockedExchangeAdd(&seen_steps, 0) < 8 && step != 64) {
        char line[96];
        int n = wsprintfA(line, "[veh] clzero loop step=%I64x span=%I64x op=%02x", step, limit - start, op);
        line[n++] = '\n'; line[n] = 0;
        OutputDebugStringA(line);
        InterlockedIncrement(&seen_steps);
    }
    if (getenv_bool("TUXBOX_VEH_NO_CLZERO_LOOP")) return 0;
    unsigned long long span = limit - start;
    for (unsigned long long at = 0; at < span; at += 65536) {
        MEMORY_BASIC_INFORMATION mbi;
        if (!VirtualQuery((void*)(uintptr_t)(start + at), &mbi, sizeof mbi) || mbi.State != MEM_COMMIT ||
            !(mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
            return 0;
    }
    if (step == 64) {
        memset((void*)(uintptr_t)start, 0, (size_t)span);
    } else {
        // CLZERO zeroes one 64-byte block per iteration, so a step larger than 64
        // leaves the bytes in between untouched.
        unsigned long long iterations = span / step;
        if (iterations > 65536) return 0;
        for (unsigned long long it = 0; it < iterations; ++it) {
            memset((void*)(uintptr_t)(start + it * step), 0, 64);
        }
    }
    unsigned long long passes = (span + step - 1) / step;
    unsigned long long exit = start + passes * step;
    ctx->Rax = exit;
    int equal = (exit == limit);
    unsigned int flags = 0x2u; // reserved bit always set
    if (equal) flags |= 0x40u;
    ctx->EFlags = (ctx->EFlags & ~0x8d5u) | flags;
    ctx->Rip += loop;
    return 1;
}

// SSE4a non-temporal scalar stores (MOVNTSS m32,xmm / MOVNTSD m64,xmm). The
// non-temporal hint is irrelevant on this host: store the low 32/64 bits and
// step over the instruction.
static int emulate_sse4a_store(const uint8_t* c, CONTEXT* ctx) {
    if (c[0] != 0xF3 && c[0] != 0xF2) return 0;
    int i = 1, rex = 0;
    if ((c[i] & 0xF0) == 0x40) { rex = c[i]; i++; }
    if (c[i] != 0x0F || c[i + 1] != 0x2B) return 0;
    uint8_t modrm = c[i + 2];
    if ((modrm >> 6) == 3) return 0;
    int src = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
    int mod = modrm >> 6, rm = modrm & 7, n = i + 3;
    uint64_t address = 0;
    if (rm == 4) {
        uint8_t sib = c[n++];
        int scale = 1 << (sib >> 6);
        int index = ((sib >> 3) & 7) | ((rex & 2) ? 8 : 0);
        int base = (sib & 7) | ((rex & 1) ? 8 : 0);
        if (((sib >> 3) & 7) != 4) address = *gpr(ctx, index) * scale;
        if ((sib & 7) == 5 && mod == 0) { address += (uint64_t)(int64_t)*(const int32_t*)(c + n); n += 4; }
        else address += *gpr(ctx, base);
    } else if (rm == 5 && mod == 0) {
        address = ctx->Rip + n + 4 + (uint64_t)(int64_t)*(const int32_t*)(c + n);
        n += 4;
    } else {
        address = *gpr(ctx, rm);
    }
    if (mod == 1) { address += (uint64_t)(int64_t)*(const int8_t*)(c + n); n += 1; }
    else if (mod == 2) { address += (uint64_t)(int64_t)*(const int32_t*)(c + n); n += 4; }
    M128A* xmm = &ctx->Xmm0;
    if (c[0] == 0xF3) *(uint32_t*)address = (uint32_t)xmm[src].Low;
    else *(uint64_t*)address = (uint64_t)xmm[src].Low;
    ctx->Rip += n;
    return n;
}

static LONG CALLBACK veh(EXCEPTION_POINTERS* ep) {
    if (!ep || !ep->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    {
        // The title's monitor thread deliberately crashes itself (write to 0 at
        // a fixed RVA) when it has seen no progress for ~30 samples. Skipping it
        // keeps a slow-but-alive title running so progress can be observed.
        DWORD code = ep->ExceptionRecord->ExceptionCode;
        CONTEXT* ctx = ep->ContextRecord;
        uintptr_t base = (uintptr_t)GetModuleHandleW(NULL);
        if (code == 0xC0000005u && (uintptr_t)ctx->Rip == base + 0x2215C35u) {
            static int once;
            if (!once) { once = 1; OutputDebugStringA("[veh] skipped title no-progress fatal assert"); }
            ctx->Rip = base + 0x2215C41u;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (code == 0x80000003u && (uintptr_t)ctx->Rip == base + 0x19F4F58u) {
            static int once2;
            if (!once2) { once2 = 1; OutputDebugStringA("[veh] skipped title debug breakpoint"); }
            ctx->Rip = base + 0x19F4F59u;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (code != (DWORD)0xC000001DL) return EXCEPTION_CONTINUE_SEARCH;
    }
    CONTEXT* ctx = ep->ContextRecord;
    uintptr_t rip = (uintptr_t)ctx->Rip;
    const uint8_t* c = (const uint8_t*)rip;
    int len = 0;
    // CLZERO 0F 01 FC
    if (c[0] == 0x0F && c[1] == 0x01 && c[2] == 0xFC) {
        if (emulate_clzero_loop(c, ctx)) { InterlockedIncrement(&hits[1]); return EXCEPTION_CONTINUE_EXECUTION; }
        uint8_t* p = (uint8_t*)(ctx->Rax & ~(uint64_t)63);
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(p, &mbi, sizeof mbi) && (mbi.Protect & (PAGE_READWRITE|PAGE_WRITECOPY|PAGE_EXECUTE_READWRITE|PAGE_EXECUTE_WRITECOPY)) && mbi.State == MEM_COMMIT) {
            memset(p, 0, 64);
        }
        {
            static int shown;
            if (!shown) {
                shown = 1;
                char b[128];
                int n = wsprintfA(b, "[veh] clzero bytes");
                for (int i = 0; i < 24 && i < 24; ++i) n += wsprintfA(b + n, " %02x", c[i]);
                b[n++] = '\n'; b[n] = 0;
                OutputDebugStringA(b);
            }
        }
        ctx->Rip += 3; InterlockedIncrement(&hits[1]); log_type(1, "clzero", 0); return EXCEPTION_CONTINUE_EXECUTION;
    }
    // MONITORX 0F 01 FA / MWAITX 0F 01 FB / RDPRU 0F 01 FD
    if (c[0] == 0x0F && c[1] == 0x01 && (c[2] == 0xFA || c[2] == 0xFB || c[2] == 0xFD)) {
        if (c[2] == 0xFD) { uint64_t t = __rdtsc(); ctx->Rax = (uint32_t)t; ctx->Rdx = (uint32_t)(t >> 32); InterlockedIncrement(&hits[4]); }
        else if (c[2] == 0xFB) { YieldProcessor(); InterlockedIncrement(&hits[3]); }
        else InterlockedIncrement(&hits[2]);
        patch_nops((void*)rip, 3); log_type(c[2] == 0xFA ? 2 : 3, c[2] == 0xFA ? "monitorx" : "mwaitx", 0); return EXCEPTION_CONTINUE_EXECUTION;
    }
    // RDPID F3 0F C7 /7
    if (c[0] == 0xF3 && c[1] == 0x0F && c[2] == 0xC7 && ((c[3] >> 3) & 7) == 7) {
        uint32_t aux = 0;
        int reg = (c[3] & 7) | ((c[3] & 0x80) ? 0 : 0);
        (void)reg;
        uint64_t* r = (uint64_t*)&ctx->Rax; // rax,rcx,rdx,rbx,rsp,rbp,rsi,rdi,...
        int idx = c[3] & 7;
        if (idx < 8) { uint64_t* tbl[8] = { &ctx->Rax,&ctx->Rcx,&ctx->Rdx,&ctx->Rbx,&ctx->Rsp,&ctx->Rbp,&ctx->Rsi,&ctx->Rdi }; *tbl[idx] = aux; }
        (void)r;
        ctx->Rip += 4; InterlockedIncrement(&hits[5]); return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (is_clwb(c, &len)) { patch_nops((void*)rip, len); InterlockedIncrement(&hits[0]); log_type(0, "clwb", len); return EXCEPTION_CONTINUE_EXECUTION; }
    if (emulate_sse4a(c, ctx)) { InterlockedIncrement(&hits[6]); return EXCEPTION_CONTINUE_EXECUTION; }
    // MOVNTSS/MOVNTSD: SSE4a non-temporal scalar stores.
    if (c[0] == 0xF3 || c[0] == 0xF2) {
        int len = emulate_sse4a_store(c, ctx);
        if (len) { InterlockedIncrement(&hits[7]); log_type(7, "ntstore", len); return EXCEPTION_CONTINUE_EXECUTION; }
        // MCOMMIT (F3 0F 01 FA): fence hint; WBNOINVD (F3 0F 09): write-back hint.
        if (c[1] == 0x0F && c[2] == 0x01 && c[3] == 0xFA) { patch_nops((void*)rip, 4); return EXCEPTION_CONTINUE_EXECUTION; }
        if (c[1] == 0x0F && c[2] == 0x09) { patch_nops((void*)rip, 3); return EXCEPTION_CONTINUE_EXECUTION; }
    }
    // PREFETCHW (0F 0D /1): a hint on this host.
    if (c[0] == 0x0F && c[1] == 0x0D && ((c[2] >> 3) & 7) == 1) {
        int n = 3, modrm = c[2], mod = modrm >> 6, rm = modrm & 7;
        if (rm == 4) { uint8_t sib = c[n++]; if ((sib & 7) == 5 && mod == 0) n += 4; }
        else if (rm == 5 && mod == 0) n += 4;
        if (mod == 1) n += 1; else if (mod == 2) n += 4;
        patch_nops((void*)rip, n);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // unknown: log once per address
    {
        static uintptr_t seen[32]; static int nseen;
        int found = 0;
        for (int i = 0; i < nseen && i < 32; i++) if (seen[i] == rip) found = 1;
        if (!found) {
            if (nseen < 32) seen[nseen++] = rip;
            char buf[128];
            wsprintfA(buf, "[veh] unhandled illegal at %p: %02x %02x %02x %02x", (void*)rip, c[0], c[1], c[2], c[3]);
            OutputDebugStringA(buf);
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

#include <tlhelp32.h>

// Freestanding: the proxy has no CRT, but the compiler may emit these.
void* __cdecl memset(void* destination, int value, size_t count) {
    unsigned char* p = (unsigned char*)destination;
    while (count--) *p++ = (unsigned char)value;
    return destination;
}
int __cdecl memcmp(const void* left, const void* right, size_t count) {
    const unsigned char* a = (const unsigned char*)left;
    const unsigned char* b = (const unsigned char*)right;
    while (count--) { if (*a != *b) return (int)*a - (int)*b; ++a; ++b; }
    return 0;
}
void* __cdecl memcpy(void* destination, const void* source, size_t count) {
    unsigned char* d = (unsigned char*)destination;
    const unsigned char* s = (const unsigned char*)source;
    while (count--) *d++ = *s++;
    return destination;
}


typedef long (__stdcall *NtGetNextThreadFn)(void* process, void* thread, unsigned long access, unsigned long flags, unsigned long zero, void** next);
typedef long (__stdcall *NtQueryInformationThreadFn)(void* thread, int cls, void* info, unsigned long len, unsigned long* ret);

static void dump_thread(void* handle, unsigned long tid) {
    if (tid == GetCurrentThreadId()) return;
    SuspendThread(handle);
    static CONTEXT c;
    c.ContextFlags = CONTEXT_CONTROL;
    if (GetThreadContext(handle, &c)) {
        char line[384];
        uintptr_t base = (uintptr_t)GetModuleHandleW(NULL);
        unsigned nf = 0;
        unsigned long long frames[16];
        uintptr_t sp = (uintptr_t)c.Rsp;
        for (int page = 0; page < 24 && nf < 16; ++page) {
            static unsigned long long buf[512];
            SIZE_T got = 0;
            if (!ReadProcessMemory(GetCurrentProcess(), (void*)(sp + page * 4096u), buf, sizeof buf, &got) || got == 0) break;
            for (SIZE_T i = 0; i < got / 8 && nf < 16; ++i) {
                uintptr_t v = (uintptr_t)buf[i];
                if (v >= base && v - base < 0x0b100000) frames[nf++] = v;
            }
        }
        int n = wsprintfA(line, "[veh] tid=%lx rip=%p", tid, (void*)c.Rip);
        for (unsigned i = 0; i < nf; ++i) n += wsprintfA(line + n, " +%lx", (unsigned long)(frames[i] - base));
        line[n++] = '\n'; line[n] = 0;
        OutputDebugStringA(line);
    }
    ResumeThread(handle);
}

static DWORD WINAPI sampler(LPVOID unused) {
    (void)unused;
    static const int delays[5] = { 30000, 60000, 90000, 120000, 150000 };
    for (int round = 0; round < 5; ++round) {
        Sleep(delays[round]);
        char head[64];
        int h = wsprintfA(head, "[veh] snapshot %d", round);
        OutputDebugStringA(head);
        unsigned count = 0;
        HMODULE ntdll = GetModuleHandleA("ntdll.dll");
        NtGetNextThreadFn next = ntdll ? (NtGetNextThreadFn)GetProcAddress(ntdll, "NtGetNextThread") : 0;
        if (next) {
            void* thread = 0;
            for (;;) {
                void* out = 0;
                long status = next(GetCurrentProcess(), thread, 0x1FFFFF, 0, 0, &out);
                if (thread) CloseHandle(thread);
                if (status != 0 || !out) break;
                thread = out;
                ++count;
                unsigned long tid = GetThreadId(thread);
                dump_thread(thread, tid);
            }
            if (thread) CloseHandle(thread);
        } else {
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snap != INVALID_HANDLE_VALUE) {
                DWORD me = GetCurrentProcessId();
                static THREADENTRY32 te;
                te.dwSize = sizeof(te);
                BOOL ok = Thread32First(snap, &te);
                while (ok) {
                    if (te.th32OwnerProcessID == me) {
                        ++count;
                        HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, te.th32ThreadID);
                        if (th) { dump_thread(th, te.th32ThreadID); CloseHandle(th); }
                    }
                    te.dwSize = sizeof(te);
                    ok = Thread32Next(snap, &te);
                }
                CloseHandle(snap);
            }
        }
        char tail[64];
        int t = wsprintfA(tail, "[veh] snapshot threads=%lu", (unsigned long)count);
        OutputDebugStringA(tail);
    }
    return 0;
}


static DWORD WINAPI self_exit_thread(void* unused) {
    (void)unused;
    char text[32];
    DWORD length = GetEnvironmentVariableA("TUXBOX_SELF_EXIT_AT_MS", text, sizeof text);
    if (length == 0 || length >= sizeof text) return 0;
    DWORD millis = 0;
    for (DWORD i = 0; i < length; ++i) {
        if (text[i] < '0' || text[i] > '9') return 0;
        millis = millis * 10 + (DWORD)(text[i] - '0');
    }
    Sleep(millis);
    // ExitProcess runs the DLL_PROCESS_DETACH handlers that merge vkd3d-proton's
    // pipeline cache; a killed process never does, so every run would recompile.
    ExitProcess(0);
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)inst; (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        if (!InterlockedExchange(&inited, 1)) {
            { HANDLE m = CreateFileA("C:\\veh-loaded.txt", GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, 0, NULL);
              if (m != INVALID_HANDLE_VALUE) { DWORD w; WriteFile(m, "loaded", 6, &w, NULL); CloseHandle(m); } }
            AddVectoredExceptionHandler(1, veh);
            OutputDebugStringA("[veh] loaded");
            {
                HANDLE s = CreateThread(NULL, 0, sampler, NULL, 0, NULL);
                if (s) CloseHandle(s);
            }
            {
                HANDLE s = CreateThread(NULL, 0, self_exit_thread, NULL, 0, NULL);
                if (s) CloseHandle(s);
            }
        }
        DisableThreadLibraryCalls(inst);
    }
    return TRUE;
}
