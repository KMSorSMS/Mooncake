// Copyright 2025 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENT_IO_URING_REACTOR_H
#define TENT_IO_URING_REACTOR_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "tent/common/concurrent/thread_pool.h"
#include "tent/common/status.h"

namespace mooncake {
namespace tent {

struct IOUringSubBatch;

// Shared epoll/eventfd reactor for io_uring completion dispatch.
//
// One reactor thread blocks in epoll_wait on the per-batch eventfds registered
// via io_uring_register_eventfd. When a batch's eventfd fires, the reactor
// hands the batch off to a worker pool task that drains all currently-ready
// CQEs in a single pass under the batch's ring_mutex.
//
// Single-flight dispatch is enforced through IOUringSubBatch::dispatch_pending,
// so at most one worker task per batch is in flight at any time.
class IOUringReactor {
   public:
    IOUringReactor() = default;
    ~IOUringReactor();

    IOUringReactor(const IOUringReactor&) = delete;
    IOUringReactor& operator=(const IOUringReactor&) = delete;

    Status start(size_t worker_threads);
    void stop();

    // Adds the batch to the epoll set keyed by its eventfd.
    Status registerBatch(IOUringSubBatch* batch);

    // Removes the batch and returns only after no reactor thread or worker
    // task can still touch it. After this call the batch's ring/eventfd may
    // safely be torn down.
    void unregisterBatch(IOUringSubBatch* batch);

   private:
    void reactorLoop();
    void dispatch(IOUringSubBatch* batch);
    static void drainCompletions(IOUringSubBatch* batch);

    int epoll_fd_ = -1;
    int control_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread reactor_thread_;
    std::unique_ptr<ThreadPool> worker_pool_;

    std::mutex registry_mutex_;
    std::unordered_map<int /*eventfd*/, IOUringSubBatch*> registry_;
};

}  // namespace tent
}  // namespace mooncake

#endif  // TENT_IO_URING_REACTOR_H
