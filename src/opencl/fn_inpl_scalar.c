__kernel void {name}(__global {T1} *A, const {T2} B) {{
    int i = get_global_id(0);
    A[i] {op} B;
}}