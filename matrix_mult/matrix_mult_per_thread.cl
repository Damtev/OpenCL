kernel void matrix_mul(global const float *first_matrix, // [N x M]
                global const float *second_matrix, // [M x L]
                global float *result_matrix, // [N x L]
                uint n, uint m, uint l) {
    const uint local_row = get_local_id(0);
    const uint local_col = get_local_id(1) * PER_THREAD;

    const uint global_row = get_global_id(0);
    const uint global_col = get_global_id(1) * PER_THREAD;

    local float a_sub[TILE_SIZE * TILE_SIZE];
    local float b_sub[TILE_SIZE * TILE_SIZE];

    float sum[PER_THREAD] = {};

    uint num_tiles = m / TILE_SIZE;
    for (uint tile = 0; tile < num_tiles; tile++) {

        // Load one tile of A and B into local memory
        for (uint work = 0; work < PER_THREAD; work++) {
            uint tiled_row = tile * TILE_SIZE + local_col + work;
            uint tiled_col = tile * TILE_SIZE + local_row;
            a_sub[(local_col + work) * TILE_SIZE + local_row] = first_matrix[(global_col + work) * m + tiled_col];
            b_sub[(local_col + work) * TILE_SIZE + local_row] = second_matrix[tiled_row * l + global_row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            for (uint work = 0; work < PER_THREAD; work++) {
                sum[work] += a_sub[(local_col + work) * TILE_SIZE + k] * b_sub[k * TILE_SIZE + local_row];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (uint work = 0; work < PER_THREAD; work++) {
        result_matrix[(global_col + work) * l + global_row] = sum[work];
    }
}