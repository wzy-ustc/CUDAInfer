#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nt {

struct KVPhysicalBlock {
    int block_id = -1;
    int offset_tokens = 0;
};

class KVBlockAllocator {
public:
    explicit KVBlockAllocator(int num_blocks = 0);

    int allocate();
    void free(int block_id);
    void retain(int block_id);
    void release(int block_id);

    int capacity() const { return capacity_; }
    int free_count() const { return static_cast<int>(free_list_.size()); }
    int used_count() const { return capacity_ - free_count(); }
    int ref_count(int block_id) const;
    bool is_allocated(int block_id) const;

private:
    void check_block_id(int block_id) const;

    int capacity_ = 0;
    std::vector<int> free_list_;
    std::vector<int> ref_counts_;
};

class SequenceBlockTable {
public:
    SequenceBlockTable() = default;
    SequenceBlockTable(int64_t seq_id, int block_tokens);

    int64_t seq_id() const { return seq_id_; }
    int block_tokens() const { return block_tokens_; }
    int num_tokens() const { return num_tokens_; }
    int num_blocks() const { return static_cast<int>(blocks_.size()); }
    const std::vector<int>& blocks() const { return blocks_; }

    void append_token(KVBlockAllocator& allocator);
    void append_tokens(KVBlockAllocator& allocator, int count);
    KVPhysicalBlock block_for_token(int token_index) const;
    void clear(KVBlockAllocator& allocator);
    void set_shared_prefix(const std::vector<int>& blocks, int prefix_tokens,
                           KVBlockAllocator& allocator);

private:
    int64_t seq_id_ = -1;
    int block_tokens_ = 0;
    int num_tokens_ = 0;
    std::vector<int> blocks_;
};

class PagedKVCache {
public:
    PagedKVCache(int num_layers, int num_kv_heads, int head_dim,
                 int block_tokens, int num_blocks);

    SequenceBlockTable& create_sequence(int64_t seq_id);
    SequenceBlockTable* find_sequence(int64_t seq_id);
    const SequenceBlockTable* find_sequence(int64_t seq_id) const;
    void release_sequence(int64_t seq_id);

    KVBlockAllocator& allocator() { return allocator_; }
    const KVBlockAllocator& allocator() const { return allocator_; }

    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    int block_tokens() const { return block_tokens_; }
    int num_blocks() const { return num_blocks_; }
    int sequence_count() const { return static_cast<int>(sequences_.size()); }
    int capacity_tokens() const { return block_tokens_ * num_blocks_; }
    size_t bytes_per_block() const;

private:
    int num_layers_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int block_tokens_ = 0;
    int num_blocks_ = 0;
    KVBlockAllocator allocator_;
    std::unordered_map<int64_t, SequenceBlockTable> sequences_;
};

} // namespace nt
