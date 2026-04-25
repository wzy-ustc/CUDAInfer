#include "request_scheduler.h"

#include <algorithm>
#include <stdexcept>

namespace nt {

ContinuousBatchingScheduler::ContinuousBatchingScheduler(int max_batch_size)
    : max_batch_size_(max_batch_size) {
    if (max_batch_size <= 0) {
        throw std::invalid_argument("max_batch_size must be positive");
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
    slot.generated_tokens = 0;
    slot.arrival_order = next_arrival_order_++;

    int64_t id = slot.request.request_id;
    requests_.emplace(id, std::move(slot));
    order_.push_back(id);
}

ScheduledBatch ContinuousBatchingScheduler::next_batch() const {
    ScheduledBatch batch;

    // Continuous batching admits waiting prefills first. This allows new requests
    // to join a running decode workload without waiting for all old requests to finish.
    for (int64_t id : order_) {
        auto it = requests_.find(id);
        if (it == requests_.end() || it->second.state != RequestState::WaitingPrefill) {
            continue;
        }
        if (batch.phase == BatchPhase::None) {
            batch.phase = BatchPhase::Prefill;
        }
        batch.request_ids.push_back(id);
        batch.token_counts.push_back(static_cast<int>(it->second.request.prompt_tokens.size()));
        if (static_cast<int>(batch.request_ids.size()) >= max_batch_size_) {
            return batch;
        }
    }
    if (!batch.empty()) {
        return batch;
    }

    batch.phase = BatchPhase::Decode;
    for (int64_t id : order_) {
        auto it = requests_.find(id);
        if (it == requests_.end() || it->second.state != RequestState::WaitingDecode) {
            continue;
        }
        batch.request_ids.push_back(id);
        batch.token_counts.push_back(1);
        if (static_cast<int>(batch.request_ids.size()) >= max_batch_size_) {
            return batch;
        }
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
    slot.state = RequestState::WaitingDecode;
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

bool ContinuousBatchingScheduler::has_pending_work() const {
    return !requests_.empty();
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
