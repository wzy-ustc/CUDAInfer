#include "request_scheduler.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace nt {

int ScheduledBatch::total_tokens() const {
    return std::accumulate(token_counts.begin(), token_counts.end(), 0);
}

ContinuousBatchingScheduler::ContinuousBatchingScheduler(int max_batch_size)
    : ContinuousBatchingScheduler(SchedulerConfig{.max_batch_size = max_batch_size}) {}

ContinuousBatchingScheduler::ContinuousBatchingScheduler(SchedulerConfig config)
    : config_(config) {
    if (config_.max_batch_size <= 0) {
        throw std::invalid_argument("max_batch_size must be positive");
    }
    if (config_.max_batched_tokens < 0) {
        throw std::invalid_argument("max_batched_tokens must be non-negative");
    }
    if (config_.prefill_chunk_tokens < 0) {
        throw std::invalid_argument("prefill_chunk_tokens must be non-negative");
    }
}

void ContinuousBatchingScheduler::enqueue(InferenceRequest request) {
    if (request.request_id < 0) {
        throw std::invalid_argument("request_id must be non-negative");
    }
    if (request.prompt_tokens.empty()) {
        throw std::invalid_argument("prompt_tokens must not be empty");
    }
    if (request.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (requests_.contains(request.request_id)) {
        throw std::runtime_error("duplicate request_id in scheduler");
    }

    RequestSlot slot;
    slot.request = std::move(request);
    slot.state = RequestState::WaitingPrefill;
    slot.prefill_tokens_done = 0;
    slot.generated_tokens = 0;
    slot.arrival_order = next_arrival_order_++;

    int64_t id = slot.request.request_id;
    requests_.emplace(id, std::move(slot));
    order_.push_back(id);
}

ScheduledBatch ContinuousBatchingScheduler::next_batch() const {
    if (config_.decode_first) {
        ScheduledBatch decode = schedule_decode();
        if (!decode.empty()) {
            return decode;
        }
        return schedule_prefill();
    }

    ScheduledBatch prefill = schedule_prefill();
    if (!prefill.empty()) {
        return prefill;
    }
    return schedule_decode();
}

ScheduledBatch ContinuousBatchingScheduler::schedule_prefill() const {
    ScheduledBatch batch;
    batch.phase = BatchPhase::Prefill;
    int token_budget_left = config_.max_batched_tokens > 0 ? config_.max_batched_tokens : INT32_MAX;

    for (int64_t id : order_) {
        auto it = requests_.find(id);
        if (it == requests_.end() || it->second.state != RequestState::WaitingPrefill) {
            continue;
        }
        if (static_cast<int>(batch.request_ids.size()) >= config_.max_batch_size) {
            break;
        }
        int chunk = prefill_chunk_size(it->second, token_budget_left);
        if (chunk <= 0) {
            break;
        }
        batch.request_ids.push_back(id);
        batch.token_counts.push_back(chunk);
        token_budget_left -= chunk;
    }

    if (batch.empty()) {
        batch.phase = BatchPhase::None;
    }
    return batch;
}

ScheduledBatch ContinuousBatchingScheduler::schedule_decode() const {
    ScheduledBatch batch;
    batch.phase = BatchPhase::Decode;
    int token_budget_left = config_.max_batched_tokens > 0 ? config_.max_batched_tokens : INT32_MAX;

    for (int64_t id : order_) {
        auto it = requests_.find(id);
        if (it == requests_.end() || it->second.state != RequestState::WaitingDecode) {
            continue;
        }
        if (static_cast<int>(batch.request_ids.size()) >= config_.max_batch_size || token_budget_left <= 0) {
            break;
        }
        batch.request_ids.push_back(id);
        batch.token_counts.push_back(1);
        --token_budget_left;
    }

    if (batch.empty()) {
        batch.phase = BatchPhase::None;
    }
    return batch;
}

void ContinuousBatchingScheduler::mark_prefill_complete(int64_t request_id) {
    RequestSlot& slot = get_slot(request_id);
    if (slot.state != RequestState::WaitingPrefill) {
        throw std::runtime_error("request is not waiting for prefill completion");
    }
    slot.prefill_tokens_done = static_cast<int>(slot.request.prompt_tokens.size());
    slot.state = RequestState::WaitingDecode;
}

void ContinuousBatchingScheduler::mark_prefill_progress(int64_t request_id, int processed_tokens) {
    if (processed_tokens <= 0) {
        throw std::invalid_argument("processed_tokens must be positive");
    }
    RequestSlot& slot = get_slot(request_id);
    if (slot.state != RequestState::WaitingPrefill) {
        throw std::runtime_error("request is not waiting for prefill progress");
    }
    int remaining = prefill_tokens_remaining(slot);
    if (processed_tokens > remaining) {
        throw std::invalid_argument("processed_tokens exceeds remaining prompt tokens");
    }
    slot.prefill_tokens_done += processed_tokens;
    if (slot.prefill_tokens_done == static_cast<int>(slot.request.prompt_tokens.size())) {
        slot.state = RequestState::WaitingDecode;
    }
}

void ContinuousBatchingScheduler::mark_decode_token(int64_t request_id, bool finished) {
    RequestSlot& slot = get_slot(request_id);
    if (slot.state != RequestState::WaitingDecode) {
        throw std::runtime_error("request is not in decode state");
    }
    ++slot.generated_tokens;
    if (finished || slot.generated_tokens >= slot.request.max_new_tokens) {
        int64_t id = slot.request.request_id;
        requests_.erase(id);
        order_.erase(std::remove(order_.begin(), order_.end(), id), order_.end());
    }
}

const InferenceRequest* ContinuousBatchingScheduler::find(int64_t request_id) const {
    auto it = requests_.find(request_id);
    return it == requests_.end() ? nullptr : &it->second.request;
}

RequestState ContinuousBatchingScheduler::state(int64_t request_id) const {
    return get_slot(request_id).state;
}

int ContinuousBatchingScheduler::generated_tokens(int64_t request_id) const {
    return get_slot(request_id).generated_tokens;
}

int ContinuousBatchingScheduler::prefill_tokens_done(int64_t request_id) const {
    return get_slot(request_id).prefill_tokens_done;
}

bool ContinuousBatchingScheduler::has_pending_work() const {
    return !requests_.empty();
}

int ContinuousBatchingScheduler::prefill_tokens_remaining(const RequestSlot& slot) const {
    return static_cast<int>(slot.request.prompt_tokens.size()) - slot.prefill_tokens_done;
}

int ContinuousBatchingScheduler::prefill_chunk_size(const RequestSlot& slot, int token_budget_left) const {
    int remaining = prefill_tokens_remaining(slot);
    int chunk = remaining;
    if (config_.prefill_chunk_tokens > 0) {
        chunk = std::min(chunk, config_.prefill_chunk_tokens);
    }
    // Do not split a configured chunk only because the global token budget has
    // one or two tokens left; tiny tail fragments hurt prefill efficiency and
    // can create scheduler churn. The final prompt tail is still allowed when it
    // is naturally smaller than prefill_chunk_tokens.
    if (chunk > token_budget_left) {
        return 0;
    }
    return chunk;
}

ContinuousBatchingScheduler::RequestSlot&
ContinuousBatchingScheduler::get_slot(int64_t request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found in scheduler");
    }
    return it->second;
}

const ContinuousBatchingScheduler::RequestSlot&
ContinuousBatchingScheduler::get_slot(int64_t request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::out_of_range("request_id not found in scheduler");
    }
    return it->second;
}

} // namespace nt
