bool tb_nan_bits(uint bits) { return (bits & 0x7fffffffu) > 0x7f800000u; }
uint tb_cvt_u32_f32(float x) {
    if (!(x > 0.0) || tb_nan_bits(asuint(x))) return 0u;
    if (x >= 4294967296.0) return 0xffffffffu;
    return uint(x);
}
uint tb_cvt_i32_f32(float x) {
    if (tb_nan_bits(asuint(x))) return 0u;
    if (x >= 2147483648.0) return 0x7fffffffu;
    if (x <= -2147483648.0) return 0x80000000u;
    return asuint(int(x));
}
