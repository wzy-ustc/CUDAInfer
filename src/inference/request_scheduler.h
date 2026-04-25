#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace nt {

enum class RequestState {
    WaitingPrefill,
    WaitingDecode,
    Finished,
};

enum class BatchPhase {
    None,
    Prefill,
    Decode,
};

struct InferenceRequest {
    int64_t request_id = -1;
    std::vector<int> prompt_tokens;
    int max_new_tokens = 0;
};

struct SchedulerConfig {
    int max_batch_size = 1;
    int max_batched_tokens = 0;
    int prefill_chunk_tokens = 0;
    bool decode_first = false;
};

struct ScheduledBatch {
    BatchPhase phase = BatchPhase::None;
    std::vector<int64_t> request_ids;
    std::vector<int> token_counts;

    bool empty() const { return request_ids.empty(); }
    int total_tokens() const;
};

class ContinuousBatchingScheduler {
public:
    explicit ContinuousBatchingScheduler(int max_batch_size);
    explicit ContinuousBatchingScheduler(SchedulerConfig config);

    void enqueue(InferenceRequest request);
    ScheduledBatch next_batch() const;
    void mark_prefill_complete(int64_t request_id);
    void mark_prefill_progress(int64_t request_id, int processed_tokens);
    void mark_decode_token(int64_t request_id, bool finished);

    const InferenceRequest* find(int64_t request_id) const;
    RequestState state(int64_t request_id) const;
    int generated_tokens(int64_t request_id) const;
    int prefill_tokens_done(int64_t request_id) const;
    bool has_pending_work() const;
    int active_count() const { return static_cast<int>(requests_.size()); }
    int max_batch_size() const { return config_.max_batch_size; }
    const SchedulerConfig& config() const { return config_; }

private:
    struct RequestSlot {
        InferenceRequest request;
        RequestState state = RequestState::WaitingPrefill;
        int prefill_tokens_done = 0;
        int generated_tokens = 0;
        uint64_t arrival_order = 0;
    };

    RequestSlot& get_slot(int64_t request_id);
    const RequestSlot& get_slot(int64_t request_id) const;
    ScheduledBatch schedule_prefill() const;
    ScheduledBatch schedule_decode() const;
    int prefill_tokens_remaining(const RequestSlot& slot) const;
    int prefill_chunk_size(const RequestSlot& slot, int token_budget_left) const;

    SchedulerConfig config_;
    uint64_t next_arrival_order_ = 0;
    std::unordered_map<int64_t, RequestSlot> requests_;
    std::deque<int64_t> order_;
};

} // namespace nt
