#include "../src/inference/request_scheduler.h"

#include <cassert>
#include <cstdio>

using namespace nt;

void test_scheduler_prefills_then_decodes_requests() {
    fprintf(stderr, "test_scheduler_prefills_then_decodes_requests...\n");

    ContinuousBatchingScheduler scheduler(/*max_batch_size=*/4);
    scheduler.enqueue({.request_id = 10, .prompt_tokens = {1, 2, 3}, .max_new_tokens = 2});
    scheduler.enqueue({.request_id = 11, .prompt_tokens = {4, 5}, .max_new_tokens = 1});

    auto first = scheduler.next_batch();
    assert(first.phase == BatchPhase::Prefill);
    assert(first.request_ids.size() == 2);
    assert(first.token_counts[0] == 3);
    assert(first.token_counts[1] == 2);

    scheduler.mark_prefill_complete(10);
    scheduler.mark_prefill_complete(11);

    auto decode = scheduler.next_batch();
    assert(decode.phase == BatchPhase::Decode);
    assert(decode.request_ids.size() == 2);
    assert(decode.token_counts[0] == 1);
    assert(decode.token_counts[1] == 1);
    fprintf(stderr, "  PASS\n");
}

void test_scheduler_removes_finished_requests_and_continues_active_ones() {
    fprintf(stderr, "test_scheduler_removes_finished_requests_and_continues_active_ones...\n");

    ContinuousBatchingScheduler scheduler(/*max_batch_size=*/2);
    scheduler.enqueue({.request_id = 1, .prompt_tokens = {7}, .max_new_tokens = 2});
    scheduler.enqueue({.request_id = 2, .prompt_tokens = {8}, .max_new_tokens = 1});
    auto prefill = scheduler.next_batch();
    assert(prefill.request_ids.size() == 2);
    scheduler.mark_prefill_complete(1);
    scheduler.mark_prefill_complete(2);

    auto step0 = scheduler.next_batch();
    assert(step0.request_ids.size() == 2);
    scheduler.mark_decode_token(1, /*finished=*/false);
    scheduler.mark_decode_token(2, /*finished=*/true);

    assert(scheduler.find(2) == nullptr);
    auto step1 = scheduler.next_batch();
    assert(step1.phase == BatchPhase::Decode);
    assert(step1.request_ids.size() == 1);
    assert(step1.request_ids[0] == 1);
    scheduler.mark_decode_token(1, /*finished=*/false);
    assert(scheduler.find(1) == nullptr);
    assert(!scheduler.has_pending_work());
    fprintf(stderr, "  PASS\n");
}

void test_scheduler_admits_new_request_while_decode_is_running() {
    fprintf(stderr, "test_scheduler_admits_new_request_while_decode_is_running...\n");

    ContinuousBatchingScheduler scheduler(/*max_batch_size=*/2);
    scheduler.enqueue({.request_id = 21, .prompt_tokens = {1}, .max_new_tokens = 3});
    auto p0 = scheduler.next_batch();
    assert(p0.phase == BatchPhase::Prefill);
    scheduler.mark_prefill_complete(21);

    auto d0 = scheduler.next_batch();
    assert(d0.phase == BatchPhase::Decode);
    scheduler.mark_decode_token(21, false);

    scheduler.enqueue({.request_id = 22, .prompt_tokens = {2, 3}, .max_new_tokens = 1});
    auto p1 = scheduler.next_batch();
    assert(p1.phase == BatchPhase::Prefill);
    assert(p1.request_ids.size() == 1);
    assert(p1.request_ids[0] == 22);
    scheduler.mark_prefill_complete(22);

    auto d1 = scheduler.next_batch();
    assert(d1.phase == BatchPhase::Decode);
    assert(d1.request_ids.size() == 2);
    fprintf(stderr, "  PASS\n");
}

int main() {
    test_scheduler_prefills_then_decodes_requests();
    test_scheduler_removes_finished_requests_and_continues_active_ones();
    test_scheduler_admits_new_request_while_decode_is_running();
    fprintf(stderr, "All continuous batching tests passed!\n");
    return 0;
}
