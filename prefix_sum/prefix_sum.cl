kernel void prefix_sum(global const float *in, global float *out, uint n){

    const uint local_id = get_local_id(0);
    const uint parts = n / PART_SIZE;

    local float even[PART_SIZE];
    local float odd[PART_SIZE];

    float last = 0.0;

    uint up_bound = (uint) log2((float) PART_SIZE);
    for (uint part = 0; part < parts; part++){
        even[local_id] = in[part * PART_SIZE + local_id];

        barrier(CLK_LOCAL_MEM_FENCE);

        uint max_id = 1;
        for (uint i = 0; i < up_bound; i++) {
            if (i % 2 == 0) {
                odd[local_id] = even[local_id] + ((local_id < max_id) ? 0.0 : even[local_id - max_id]);
            } else {
                even[local_id] = odd[local_id] + ((local_id < max_id) ? 0.0 : odd[local_id - max_id]);
            }

            max_id <<= 1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        out[part * PART_SIZE + local_id] = last;
        if (up_bound - 1 == 0) {
             out[part * PART_SIZE + local_id] += odd[local_id];
             last += odd[PART_SIZE - 1];
        } else {
             out[part * PART_SIZE + local_id] += even[local_id];
             last += even[PART_SIZE - 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}