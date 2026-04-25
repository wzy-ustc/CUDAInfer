#include "prefix_kv_cache.h"

#include <sstream>
#include <stdexcept>

namespace nt {

PrefixKVCache::PrefixKVCache(KVBlockAllocator& allocator) : allocator_(allocator) {}

PrefixKVCache::~PrefixKVCache() {
    clear();
}

std::string PrefixKVCache::insert(const std::vector<int>& tokens,
                                  const SequenceBlockTable& table) {
    if (tokens.empty()) {
        throw std::invalid_argument("prefix tokens must not be empty");
    }
    if (table.num_tokens() != static_cast<int>(tokens.size())) {
        throw std::invalid_argument("prefix token count must match sequence table token count");
    }

    std::string key = hash_tokens(tokens);
    if (entries_.contains(key)) {
        erase(key);
    }

    Entry entry;
    entry.tokens = tokens;
    entry.blocks = table.blocks();
    for (int block_id : entry.blocks) {
        allocator_.retain(block_id);
    }
    entries_.emplace(key, std::move(entry));
    return key;
}

std::optional<PrefixCacheMatch>
PrefixKVCache::longest_prefix_match(const std::vector<int>& tokens) const {
    const Entry* best = nullptr;
    std::string best_key;
    for (const auto& [key, entry] : entries_) {
        if (!is_prefix(entry, tokens)) {
            continue;
        }
        if (best == nullptr || entry.tokens.size() > best->tokens.size()) {
            best = &entry;
            best_key = key;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    return PrefixCacheMatch{
        .key = best_key,
        .tokens = static_cast<int>(best->tokens.size()),
        .blocks = best->blocks,
    };
}

void PrefixKVCache::attach_to_sequence(const PrefixCacheMatch& match,
                                       SequenceBlockTable& target) {
    target.set_shared_prefix(match.blocks, match.tokens, allocator_);
}

bool PrefixKVCache::erase(const std::string& key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        return false;
    }
    for (int block_id : it->second.blocks) {
        allocator_.release(block_id);
    }
    entries_.erase(it);
    return true;
}

void PrefixKVCache::clear() {
    for (auto& [_, entry] : entries_) {
        for (int block_id : entry.blocks) {
            allocator_.release(block_id);
        }
    }
    entries_.clear();
}

std::string PrefixKVCache::hash_tokens(const std::vector<int>& tokens) {
    // Stable process-independent key. This is deliberately not std::hash so
    // cached prefixes can later be serialized or logged deterministically.
    uint64_t h = 1469598103934665603ull; // FNV-1a offset
    for (int token : tokens) {
        uint32_t v = static_cast<uint32_t>(token);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>((v >> (i * 8)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    std::ostringstream oss;
    oss << tokens.size() << ':' << std::hex << h;
    return oss.str();
}

bool PrefixKVCache::is_prefix(const Entry& entry, const std::vector<int>& tokens) const {
    if (entry.tokens.size() > tokens.size()) {
        return false;
    }
    for (size_t i = 0; i < entry.tokens.size(); ++i) {
        if (entry.tokens[i] != tokens[i]) {
            return false;
        }
    }
    return true;
}

} // namespace nt
