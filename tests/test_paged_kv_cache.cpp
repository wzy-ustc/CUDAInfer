#include "../src/inference/paged_kv_cache.h"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace nt;

void test_block_allocator_reuses_freed_blocks() {
    fprintf(stderr, "test_block_allocator_reuses_freed_blocks...\n");

    KVBlockAllocator allocator(4);
    int a = allocator.allocate();
    int b = allocator.allocate();
    assert(a == 0);
    assert(b == 1);
    assert(allocator.free_count() == 2);

    allocator.free(a);
    assert(allocator.free_count() == 3);

    int c = allocator.allocate();
    assert(c == a);
    assert(allocator.free_count() == 2);
    fprintf(stderr, "  PASS\n");
}

void test_sequence_block_table_grows_by_token_blocks() {
    fprintf(stderr, "test_sequence_block_table_grows_by_token_blocks...\n");

    PagedKVCache cache(/*num_layers=*/2, /*num_kv_heads=*/4, /*head_dim=*/8,
                       /*block_tokens=*/4, /*num_blocks=*/8);
    auto& table = cache.create_sequence(42);

    for (int pos = 0; pos < 9; ++pos) {
        table.append_token(cache.allocator());
    }

    assert(table.seq_id() == 42);
    assert(table.num_tokens() == 9);
    assert(table.num_blocks() == 3);
    assert(table.block_for_token(0).block_id == 0);
    assert(table.block_for_token(3).block_id == 0);
    assert(table.block_for_token(4).block_id == 1);
    assert(table.block_for_token(8).block_id == 2);
    assert(cache.allocator().free_count() == 5);
    fprintf(stderr, "  PASS\n");
}

void test_sequence_release_returns_blocks_to_allocator() {
    fprintf(stderr, "test_sequence_release_returns_blocks_to_allocator...\n");

    PagedKVCache cache(/*num_layers=*/1, /*num_kv_heads=*/2, /*head_dim=*/16,
                       /*block_tokens=*/2, /*num_blocks=*/4);
    auto& table = cache.create_sequence(7);
    table.append_tokens(cache.allocator(), 5);
    assert(table.num_blocks() == 3);
    assert(cache.allocator().free_count() == 1);

    cache.release_sequence(7);
    assert(cache.sequence_count() == 0);
    assert(cache.allocator().free_count() == 4);
    fprintf(stderr, "  PASS\n");
}

void test_cache_reports_block_bytes_for_f16_kv() {
    fprintf(stderr, "test_cache_reports_block_bytes_for_f16_kv...\n");

    PagedKVCache cache(/*num_layers=*/3, /*num_kv_heads=*/8, /*head_dim=*/16,
                       /*block_tokens=*/4, /*num_blocks=*/10);
    const size_t expected = 2u * 3u * 8u * 16u * 4u * sizeof(uint16_t);
    assert(cache.bytes_per_block() == expected);
    assert(cache.capacity_tokens() == 40);
    fprintf(stderr, "  PASS\n");
}

int main() {
    test_block_allocator_reuses_freed_blocks();
    test_sequence_block_table_grows_by_token_blocks();
    test_sequence_release_returns_blocks_to_allocator();
    test_cache_reports_block_bytes_for_f16_kv();
    fprintf(stderr, "All paged KV cache tests passed!\n");
    return 0;
}
