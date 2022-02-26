__kernel void {name}(__global {T1} *A, __global const {T2} *B) {{
    int i = get_global_id(0);
    A[i] {op} B[i];
}}