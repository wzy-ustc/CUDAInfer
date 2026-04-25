#pragma once

#include "paged_kv_cache.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace nt {

struct PrefixCacheMatch {
    std::string key;
    int tokens = 0;
    std::vector<int> blocks;
};

class PrefixKVCache {
public:
    explicit PrefixKVCache(KVBlockAllocator& allocator);
    ~PrefixKVCache();

    PrefixKVCache(const PrefixKVCache&) = delete;
    PrefixKVCache& operator=(const PrefixKVCache&) = delete;

    std::string insert(const std::vector<int>& tokens, const SequenceBlockTable& table);
    std::optional<PrefixCacheMatch> longest_prefix_match(const std::vector<int>& tokens) const;
    void attach_to_sequence(const PrefixCacheMatch& match, SequenceBlockTable& target);
    bool erase(const std::string& key);
    void clear();

    int entry_count() const { return static_cast<int>(entries_.size()); }
    static std::string hash_tokens(const std::vector<int>& tokens);

private:
    struct Entry {
        std::vector<int> tokens;
        std::vector<int> blocks;
    };

    bool is_prefix(const Entry& entry, const std::vector<int>& tokens) const;

    KVBlockAllocator& allocator_;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace nt
