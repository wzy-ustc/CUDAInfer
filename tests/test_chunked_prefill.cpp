#include "../src/inference/request_scheduler.h"

#include <cassert>
#include <cstdio>

using namespace nt;

void test_decode_first_policy_prioritizes_active_decode_over_new_prefill() {
    fprintf(stderr, "test_decode_first_policy_prioritizes_active_decode_over_new_prefill...\n");

    SchedulerConfig config;
    config.max_batch_size = 4;
    config.max_batched_tokens = 8;
    config.prefill_chunk_tokens = 4;
    config.decode_first = true;
    ContinuousBatchingScheduler scheduler(config);

    scheduler.enqueue({.request_id = 1, .prompt_tokens = {1, 2}, .max_new_tokens = 3});
    auto p0 = scheduler.next_batch();
    assert(p0.phase == BatchPhase::Prefill);
    scheduler.mark_prefill_progress(1, p0.token_counts[0]);

    scheduler.enqueue({.request_id = 2, .prompt_tokens = {3, 4, 5, 6, 7, 8}, .max_new_tokens = 1});
    auto d0 = scheduler.next_batch();
    assert(d0.phase == BatchPhase::Decode);
    assert(d0.request_ids.size() == 1);
    assert(d0.request_ids[0] == 1);
    fprintf(stderr, "  PASS\n");
}

void test_chunked_prefill_splits_long_prompt_by_chunk_size() {
    fprintf(stderr, "test_chunked_prefill_splits_long_prompt_by_chunk_size...\n");

    SchedulerConfig config;
    config.max_batch_size = 2;
    config.max_batched_tokens = 16;
    config.prefill_chunk_tokens = 3;
    config.decode_first = true;
    ContinuousBatchingScheduler scheduler(config);
    scheduler.enqueue({.request_id = 9, .prompt_tokens = {0, 1, 2, 3, 4, 5, 6}, .max_new_tokens = 1});

    auto c0 = scheduler.next_batch();
    assert(c0.phase == BatchPhase::Prefill);
    assert(c0.token_counts[0] == 3);
    scheduler.mark_prefill_progress(9, c0.token_counts[0]);
    assert(scheduler.prefill_tokens_done(9) == 3);
    assert(scheduler.state(9) == RequestState::WaitingPrefill);

    auto c1 = scheduler.next_batch();
    assert(c1.token_counts[0] == 3);
    scheduler.mark_prefill_progress(9, c1.token_counts[0]);
    assert(scheduler.prefill_tokens_done(9) == 6);

    auto c2 = scheduler.next_batch();
    assert(c2.token_counts[0] == 1);
    scheduler.mark_prefill_progress(9, c2.token_counts[0]);
    assert(scheduler.state(9) == RequestState::WaitingDecode);
    fprintf(stderr, "  PASS\n");
}

void test_prefill_token_budget_batches_multiple_chunks_without_exceeding_budget() {
    fprintf(stderr, "test_prefill_token_budget_batches_multiple_chunks_without_exceeding_budget...\n");

    SchedulerConfig config;
    config.max_batch_size = 4;
    config.max_batched_tokens = 5;
    config.prefill_chunk_tokens = 4;
    ContinuousBatchingScheduler scheduler(config);
    scheduler.enqueue({.request_id = 31, .prompt_tokens = {1, 2, 3, 4}, .max_new_tokens = 1});
    scheduler.enqueue({.request_id = 32, .prompt_tokens = {5, 6, 7, 8}, .max_new_tokens = 1});

    auto batch = scheduler.next_batch();
    assert(batch.phase == BatchPhase::Prefill);
    assert(batch.request_ids.size() == 1);
    assert(batch.request_ids[0] == 31);
    assert(batch.token_counts[0] == 4);
    fprintf(stderr, "  PASS\n");
}

int main() {
    test_decode_first_policy_prioritizes_active_decode_over_new_prefill();
    test_chunked_prefill_splits_long_prompt_by_chunk_size();
    test_prefill_token_budget_batches_multiple_chunks_without_exceeding_budget();
    fprintf(stderr, "All chunked prefill tests passed!\n");
    return 0;
}
