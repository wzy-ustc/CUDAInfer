#include "../src/inference/prefix_kv_cache.h"

#include <cassert>
#include <cstdio>

using namespace nt;

void test_prefix_cache_reuses_shared_blocks_with_ref_counts() {
    fprintf(stderr, "test_prefix_cache_reuses_shared_blocks_with_ref_counts...\n");

    PagedKVCache kv(/*num_layers=*/1, /*num_kv_heads=*/2, /*head_dim=*/8,
                    /*block_tokens=*/2, /*num_blocks=*/8);
    PrefixKVCache prefix(kv.allocator());

    auto& source = kv.create_sequence(100);
    source.append_tokens(kv.allocator(), 4);
    std::vector<int> sys_prompt = {10, 11, 12, 13};
    prefix.insert(sys_prompt, source);

    assert(prefix.entry_count() == 1);
    assert(kv.allocator().ref_count(source.blocks()[0]) == 2); // source + cache
    assert(kv.allocator().ref_count(source.blocks()[1]) == 2);

    auto& target = kv.create_sequence(101);
    auto match = prefix.longest_prefix_match({10, 11, 12, 13, 99});
    assert(match.has_value());
    assert(match->tokens == 4);
    prefix.attach_to_sequence(*match, target);

    assert(target.num_tokens() == 4);
    assert(target.blocks()[0] == source.blocks()[0]);
    assert(kv.allocator().ref_count(source.blocks()[0]) == 3); // source + cache + target
    fprintf(stderr, "  PASS\n");
}

void test_prefix_cache_release_does_not_free_blocks_while_sequence_uses_them() {
    fprintf(stderr, "test_prefix_cache_release_does_not_free_blocks_while_sequence_uses_them...\n");

    PagedKVCache kv(1, 1, 4, 2, 4);
    PrefixKVCache prefix(kv.allocator());
    auto& source = kv.create_sequence(1);
    source.append_tokens(kv.allocator(), 2);
    int block = source.blocks()[0];

    prefix.insert({1, 2}, source);
    auto& target = kv.create_sequence(2);
    auto match = prefix.longest_prefix_match({1, 2, 3});
    prefix.attach_to_sequence(*match, target);
    assert(kv.allocator().ref_count(block) == 3);

    prefix.erase(match->key);
    assert(kv.allocator().ref_count(block) == 2);
    kv.release_sequence(1);
    assert(kv.allocator().ref_count(block) == 1);
    kv.release_sequence(2);
    assert(!kv.allocator().is_allocated(block));
    fprintf(stderr, "  PASS\n");
}

void test_prefix_cache_prefers_longest_prefix() {
    fprintf(stderr, "test_prefix_cache_prefers_longest_prefix...\n");

    PagedKVCache kv(1, 1, 4, 2, 8);
    PrefixKVCache prefix(kv.allocator());

    auto& short_seq = kv.create_sequence(10);
    short_seq.append_tokens(kv.allocator(), 2);
    prefix.insert({7, 8}, short_seq);

    auto& long_seq = kv.create_sequence(11);
    long_seq.append_tokens(kv.allocator(), 4);
    prefix.insert({7, 8, 9, 10}, long_seq);

    auto match = prefix.longest_prefix_match({7, 8, 9, 10, 11});
    assert(match.has_value());
    assert(match->tokens == 4);
    fprintf(stderr, "  PASS\n");
}

int main() {
    test_prefix_cache_reuses_shared_blocks_with_ref_counts();
    test_prefix_cache_release_does_not_free_blocks_while_sequence_uses_them();
    test_prefix_cache_prefers_longest_prefix();
    fprintf(stderr, "All prefix KV cache tests passed!\n");
    return 0;
}
