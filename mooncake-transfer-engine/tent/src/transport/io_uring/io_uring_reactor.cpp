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

#include "tent/transport/io_uring/io_uring_reactor.h"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#include <glog/logging.h>

#include "tent/transport/io_uring/io_uring_transport.h"

namespace mooncake {
namespace tent {

namespace {
constexpr int kMaxEpollEvents = 64;
constexpr auto kBarrierPollInterval = std::chrono::microseconds(50);
constexpr auto kBarrierTimeout = std::chrono::seconds(5);
}  // namespace

IOUringReactor::~IOUringReactor() { stop(); }

Status IOUringReactor::start(size_t worker_threads) {
    if (running_.load(std::memory_order_acquire)) {
        return Status::InvalidArgument("IOUringReactor already started" LOC_MARK);
    }
    if (worker_threads == 0) worker_threads = 1;

    epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
    if (epoll_fd_ < 0) {
        return Status::InternalError(
            std::string("epoll_create1 failed: ") + strerror(errno) + LOC_MARK);
    }
    control_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (control_fd_ < 0) {
        ::close(epoll_fd_);
        epoll_fd_ = -1;
        return Status::InternalError(
            std::string("control eventfd failed: ") + strerror(errno) + LOC_MARK);
    }
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = control_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, control_fd_, &ev) < 0) {
        ::close(control_fd_);
        ::close(epoll_fd_);
        control_fd_ = -1;
        epoll_fd_ = -1;
        return Status::InternalError(
            std::string("epoll_ctl(control) failed: ") + strerror(errno) +
            LOC_MARK);
    }

    worker_pool_ = std::make_unique<ThreadPool>(worker_threads);
    running_.store(true, std::memory_order_release);
    reactor_thread_ = std::thread([this] { reactorLoop(); });
    return Status::OK();
}

void IOUringReactor::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    // Wake the reactor thread out of epoll_wait.
    if (control_fd_ >= 0) {
        const uint64_t v = 1;
        ssize_t n = ::write(control_fd_, &v, sizeof(v));
        (void)n;
    }
    if (reactor_thread_.joinable()) reactor_thread_.join();
    // Tear down workers after the reactor is fully stopped so no fresh tasks
    // can be enqueued onto a destroying pool.
    worker_pool_.reset();
    if (control_fd_ >= 0) {
        ::close(control_fd_);
        control_fd_ = -1;
    }
    if (epoll_fd_ >= 0) {
        ::close(epoll_fd_);
        epoll_fd_ = -1;
    }
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        registry_.clear();
    }
}

Status IOUringReactor::registerBatch(IOUringSubBatch* batch) {
    if (!batch) return Status::InvalidArgument("null batch" LOC_MARK);
    if (!running_.load(std::memory_order_acquire)) {
        return Status::InvalidArgument("reactor not running" LOC_MARK);
    }
    if (batch->eventfd_ < 0) {
        return Status::InvalidArgument("batch has no eventfd" LOC_MARK);
    }
    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        registry_[batch->eventfd_] = batch;
    }
    epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = batch->eventfd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, batch->eventfd_, &ev) < 0) {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        registry_.erase(batch->eventfd_);
        return Status::InternalError(
            std::string("epoll_ctl(ADD) failed: ") + strerror(errno) + LOC_MARK);
    }
    batch->registered.store(true, std::memory_order_release);
    return Status::OK();
}

bool IOUringReactor::unregisterBatch(IOUringSubBatch* batch) {
    if (!batch) return true;
    if (!batch->registered.load(std::memory_order_acquire)) return true;

    {
        std::lock_guard<std::mutex> lk(registry_mutex_);
        registry_.erase(batch->eventfd_);
    }
    if (epoll_fd_ >= 0 && batch->eventfd_ >= 0) {
        // EBADF / ENOENT are tolerable (already removed).
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, batch->eventfd_, nullptr);
    }

    // Barrier: after the registry erase no new worker can be dispatched, so
    // active_workers only decreases. Zero means no reactor or worker thread
    // can still touch this batch -- including its notifyTerminal call.
    const auto deadline = std::chrono::steady_clock::now() + kBarrierTimeout;
    while (batch->active_workers.load(std::memory_order_acquire) > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            LOG(ERROR) << "unregisterBatch: worker barrier timed out; "
                          "refusing to quiesce";
            return false;
        }
        std::this_thread::sleep_for(kBarrierPollInterval);
    }
    batch->registered.store(false, std::memory_order_release);
    return true;
}

void IOUringReactor::reactorLoop() {
    epoll_event events[kMaxEpollEvents];
    while (running_.load(std::memory_order_acquire)) {
        const int n = epoll_wait(epoll_fd_, events, kMaxEpollEvents, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            LOG(ERROR) << "epoll_wait failed: " << strerror(errno);
            break;
        }
        for (int i = 0; i < n; ++i) {
            const int fd = events[i].data.fd;
            if (fd == control_fd_) {
                uint64_t v = 0;
                ssize_t r = ::read(control_fd_, &v, sizeof(v));
                (void)r;
                continue;
            }
            // Resolve the batch through the registry and dispatch under the
            // registry lock: once unregisterBatch has erased the entry, the
            // reactor can no longer reach the batch, even via events that were
            // already returned by an earlier epoll_wait.
            std::lock_guard<std::mutex> lk(registry_mutex_);
            auto it = registry_.find(fd);
            if (it == registry_.end()) continue;  // unregistered concurrently
            IOUringSubBatch* batch = it->second;
            uint64_t v = 0;
            ssize_t r = ::read(fd, &v, sizeof(v));  // consume the signal
            (void)r;
            dispatch(batch);
        }
    }
}

void IOUringReactor::dispatch(IOUringSubBatch* batch) {
    bool expected = false;
    if (!batch->dispatch_pending.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
        // A worker is already draining this batch. It re-checks the CQ after
        // clearing dispatch_pending, so a CQE that races this failed CAS is
        // picked up there -- the consumed eventfd signal is not lost.
        return;
    }
    batch->active_workers.fetch_add(1, std::memory_order_acq_rel);
    try {
        worker_pool_->enqueue([batch] { drainCompletions(batch); });
    } catch (const std::exception& e) {
        LOG(ERROR) << "worker_pool enqueue failed: " << e.what();
        batch->dispatch_pending.store(false, std::memory_order_release);
        batch->active_workers.fetch_sub(1, std::memory_order_acq_rel);
    }
}

// noexcept: an escaping exception would strand dispatch_pending and
// active_workers, wedging the batch forever; under OOM, terminating
// beats a silent wedge.
void IOUringReactor::drainCompletions(IOUringSubBatch* batch) noexcept {
    std::vector<IOUringTransport::HarvestedCqe> done;
    for (;;) {
        done.clear();
        {
            std::lock_guard<std::mutex> lk(batch->ring_mutex);
            IOUringTransport::harvestCompletionsLocked(batch, done);
        }
        // Finalize outside ring_mutex: consumed CQEs are no longer the
        // kernel's, and the READ copy-back can be megabytes. active_workers
        // keeps the batch alive until the bottom of this function, so
        // unregisterBatch's barrier still covers everything here.
        bool any_terminal = false;
        for (const auto& h : done) {
            if (IOUringTransport::finalizeCompletion(batch, h)) {
                any_terminal = true;
            }
        }
        if (any_terminal) batch->notifyTerminal();

        // Single-flight handoff: clear the flag, then re-check the CQ. A CQE
        // whose eventfd signal the reactor consumed while our flag was still
        // set would otherwise strand -- it is in the CQ by now, so this re-peek
        // sees it. If a new dispatch already claimed the flag, it owns the CQ.
        batch->dispatch_pending.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(batch->ring_mutex);
            struct io_uring_cqe* cqe = nullptr;
            if (io_uring_peek_cqe(&batch->ring, &cqe) != 0 || !cqe) break;
        }
        bool expected = false;
        if (!batch->dispatch_pending.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel)) {
            break;
        }
    }
    batch->active_workers.fetch_sub(1, std::memory_order_acq_rel);  // last touch
}

}  // namespace tent
}  // namespace mooncake
