#include "paged_kv_cache.h"

#include <algorithm>

namespace nt {

KVBlockAllocator::KVBlockAllocator(int num_blocks)
    : capacity_(num_blocks), ref_counts_(num_blocks, 0) {
    if (num_blocks < 0) {
        throw std::invalid_argument("KVBlockAllocator num_blocks must be non-negative");
    }
    free_list_.reserve(num_blocks);
    // LIFO free list, but allocate lowest id first for deterministic tests.
    for (int id = num_blocks - 1; id >= 0; --id) {
        free_list_.push_back(id);
    }
}

int KVBlockAllocator::allocate() {
    if (free_list_.empty()) {
        throw std::runtime_error("Paged KV cache exhausted");
    }
    int block_id = free_list_.back();
    free_list_.pop_back();
    ref_counts_[block_id] = 1;
    return block_id;
}

void KVBlockAllocator::free(int block_id) {
    check_block_id(block_id);
    if (ref_counts_[block_id] == 0) {
        throw std::runtime_error("double free of KV block");
    }
    ref_counts_[block_id] = 0;
    free_list_.push_back(block_id);
}

void KVBlockAllocator::retain(int block_id) {
    check_block_id(block_id);
    if (ref_counts_[block_id] == 0) {
        throw std::runtime_error("cannot retain a free KV block");
    }
    ++ref_counts_[block_id];
}

void KVBlockAllocator::release(int block_id) {
    check_block_id(block_id);
    if (ref_counts_[block_id] == 0) {
        throw std::runtime_error("cannot release a free KV block");
    }
    --ref_counts_[block_id];
    if (ref_counts_[block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

int KVBlockAllocator::ref_count(int block_id) const {
    check_block_id(block_id);
    return ref_counts_[block_id];
}

bool KVBlockAllocator::is_allocated(int block_id) const {
    check_block_id(block_id);
    return ref_counts_[block_id] > 0;
}

void KVBlockAllocator::check_block_id(int block_id) const {
    if (block_id < 0 || block_id >= capacity_) {
        throw std::out_of_range("KV block id out of range");
    }
}

SequenceBlockTable::SequenceBlockTable(int64_t seq_id, int block_tokens)
    : seq_id_(seq_id), block_tokens_(block_tokens) {
    if (block_tokens <= 0) {
        throw std::invalid_argument("block_tokens must be positive");
    }
}

void SequenceBlockTable::append_token(KVBlockAllocator& allocator) {
    if (num_tokens_ % block_tokens_ == 0) {
        blocks_.push_back(allocator.allocate());
    }
    ++num_tokens_;
}

void SequenceBlockTable::append_tokens(KVBlockAllocator& allocator, int count) {
    if (count < 0) {
        throw std::invalid_argument("cannot append a negative token count");
    }
    for (int i = 0; i < count; ++i) {
        append_token(allocator);
    }
}

KVPhysicalBlock SequenceBlockTable::block_for_token(int token_index) const {
    if (token_index < 0 || token_index >= num_tokens_) {
        throw std::out_of_range("token index out of sequence range");
    }
    int logical_block = token_index / block_tokens_;
    return KVPhysicalBlock{blocks_.at(logical_block), token_index % block_tokens_};
}

void SequenceBlockTable::clear(KVBlockAllocator& allocator) {
    for (int block_id : blocks_) {
        allocator.release(block_id);
    }
    blocks_.clear();
    num_tokens_ = 0;
}

void SequenceBlockTable::set_shared_prefix(const std::vector<int>& blocks,
                                           int prefix_tokens,
                                           KVBlockAllocator& allocator) {
    if (prefix_tokens < 0) {
        throw std::invalid_argument("prefix_tokens must be non-negative");
    }
    if (prefix_tokens > 0) {
        int required_blocks = (prefix_tokens + block_tokens_ - 1) / block_tokens_;
        if (required_blocks != static_cast<int>(blocks.size())) {
            throw std::invalid_argument("prefix block count does not cover prefix tokens");
        }
    } else if (!blocks.empty()) {
        throw std::invalid_argument("zero-token prefix cannot have blocks");
    }
    clear(allocator);
    blocks_ = blocks;
    num_tokens_ = prefix_tokens;
    for (int block_id : blocks_) {
        allocator.retain(block_id);
    }
}

PagedKVCache::PagedKVCache(int num_layers, int num_kv_heads, int head_dim,
                           int block_tokens, int num_blocks)
    : num_layers_(num_layers), num_kv_heads_(num_kv_heads), head_dim_(head_dim),
      block_tokens_(block_tokens), num_blocks_(num_blocks), allocator_(num_blocks) {
    if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        block_tokens <= 0 || num_blocks < 0) {
        throw std::invalid_argument("invalid paged KV cache dimensions");
    }
}

SequenceBlockTable& PagedKVCache::create_sequence(int64_t seq_id) {
    auto [it, inserted] = sequences_.emplace(seq_id, SequenceBlockTable(seq_id, block_tokens_));
    if (!inserted) {
        throw std::runtime_error("sequence already exists in paged KV cache");
    }
    return it->second;
}

SequenceBlockTable* PagedKVCache::find_sequence(int64_t seq_id) {
    auto it = sequences_.find(seq_id);
    return it == sequences_.end() ? nullptr : &it->second;
}

const SequenceBlockTable* PagedKVCache::find_sequence(int64_t seq_id) const {
    auto it = sequences_.find(seq_id);
    return it == sequences_.end() ? nullptr : &it->second;
}

void PagedKVCache::release_sequence(int64_t seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) {
        return;
    }
    it->second.clear(allocator_);
    sequences_.erase(it);
}

size_t PagedKVCache::bytes_per_block() const {
    // F16 K + F16 V for every layer, KV head, token, and head element.
    return static_cast<size_t>(2) * static_cast<size_t>(num_layers_) *
           static_cast<size_t>(num_kv_heads_) * static_cast<size_t>(head_dim_) *
           static_cast<size_t>(block_tokens_) * sizeof(uint16_t);
}

} // namespace nt
