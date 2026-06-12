// Copyright 2024 KVCache.AI
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

#include "tent/transport/io_uring/io_uring_transport.h"

#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <glog/logging.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <thread>

#include "tent/runtime/slab.h"
#include "tent/common/utils/os.h"
#include "tent/runtime/platform.h"
#include "tent/transport/io_uring/io_uring_reactor.h"

namespace mooncake {
namespace tent {
class IOUringFileContext {
   public:
    explicit IOUringFileContext(const std::string& path) : ready_(false) {
        fd_ = open(path.c_str(), O_RDWR | O_DIRECT);
        if (fd_ >= 0) {
            ready_ = true;
            return;
        }

        fd_ = open(path.c_str(), O_RDWR);
        if (fd_ < 0) {
            PLOG(ERROR) << "Failed to open file " << path;
            return;
        }

        LOG(WARNING) << "File " << path << " opened in Buffered I/O mode";
        ready_ = true;
    }

    IOUringFileContext(const IOUringFileContext&) = delete;
    IOUringFileContext& operator=(const IOUringFileContext&) = delete;

    ~IOUringFileContext() {
        if (fd_ >= 0) close(fd_);
    }

    int getHandle() const { return fd_; }

    bool ready() const { return ready_; }

   private:
    int fd_;
    bool ready_;
};

IOUringTransport::IOUringTransport() : installed_(false) {}

IOUringTransport::~IOUringTransport() { uninstall(); }

Status IOUringTransport::install(std::string& local_segment_name,
                                 std::shared_ptr<ControlService> metadata,
                                 std::shared_ptr<Topology> local_topology,
                                 std::shared_ptr<Config> conf) {
    if (installed_) {
        return Status::InvalidArgument(
            "IO Uring transport has been installed" LOC_MARK);
    }

    CHECK_STATUS(probeCapabilities());
    metadata_ = metadata;
    local_segment_name_ = local_segment_name;
    local_topology_ = local_topology;
    conf_ = conf;

    // CQE draining is microsecond-scale work per entry; the only heavy part
    // is the bounce-buffer copy-back, so a small pool suffices. The reactor
    // and its workers start lazily on first allocateSubBatch so a process
    // that never issues io_uring transfers pays no thread footprint.
    constexpr int kDefaultReactorWorkers = 2;
    reactor_workers_ =
        conf_ ? conf_->get<int>("transports/io_uring/reactor_workers",
                                kDefaultReactorWorkers)
              : kDefaultReactorWorkers;

    installed_ = true;
    caps.dram_to_file = true;
    if (Platform::getLoader().type() != "cpu") {
        caps.gpu_to_file = true;
    }
    return Status::OK();
}

Status IOUringTransport::probeCapabilities() {
    struct io_uring probe_ring;
    int rc = io_uring_queue_init(2, &probe_ring, 0);
    if (rc < 0) {
        LOG(INFO) << "IOUringTransport: io_uring_queue_init failed: "
                  << strerror(-rc);
        return Status::InternalError("io_uring not supported on this kernel");
    }
    io_uring_queue_exit(&probe_ring);
    return Status::OK();
}

Status IOUringTransport::uninstall() {
    if (installed_) {
        drain();
        if (reactor_) {
            reactor_->stop();
            reactor_.reset();
        }
        metadata_.reset();
        installed_ = false;
    }
    return Status::OK();
}

Status IOUringTransport::drain() {
    std::vector<IOUringSubBatch*> snapshot;
    {
        std::lock_guard<std::mutex> lock(allocated_batches_mutex_);
        snapshot.assign(allocated_batches_.begin(), allocated_batches_.end());
    }
    constexpr auto kDrainTimeout = std::chrono::seconds(5);
    for (auto* batch : snapshot) {
        const auto deadline = std::chrono::steady_clock::now() + kDrainTimeout;
        while (batch->pending_cqes.load(std::memory_order_acquire) > 0 &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (batch->pending_cqes.load(std::memory_order_acquire) > 0) {
            LOG(WARNING) << "drain: batch still has "
                         << batch->pending_cqes.load(std::memory_order_acquire)
                         << " in-flight CQEs after timeout";
        }
    }
    return Status::OK();
}

Status IOUringTransport::ensureReactorStarted() {
    std::lock_guard<std::mutex> lk(reactor_mutex_);
    if (reactor_) return Status::OK();
    auto reactor = std::make_unique<IOUringReactor>();
    auto rs = reactor->start(static_cast<size_t>(reactor_workers_));
    if (!rs.ok()) return rs;   // retryable on the next allocate
    reactor_ = std::move(reactor);
    return Status::OK();
}

Status IOUringTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
    auto rs = ensureReactorStarted();
    if (!rs.ok()) return rs;

    auto io_uring_batch = Slab<IOUringSubBatch>::Get().allocate();
    if (!io_uring_batch)
        return Status::InternalError("Unable to allocate IO Uring sub-batch");
    batch = io_uring_batch;
    io_uring_batch->max_size = max_size;
    io_uring_batch->task_list.reserve(max_size);
    int rc = io_uring_queue_init(max_size, &io_uring_batch->ring, 0);
    if (rc) {
        Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
        batch = nullptr;
        return Status::InternalError(
            std::string("io_uring_queue_init failed: ") + strerror(-rc) +
            LOC_MARK);
    }
    io_uring_batch->eventfd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (io_uring_batch->eventfd_ < 0) {
        io_uring_queue_exit(&io_uring_batch->ring);
        Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
        batch = nullptr;
        return Status::InternalError(
            std::string("eventfd failed: ") + strerror(errno) + LOC_MARK);
    }
    rc = io_uring_register_eventfd(&io_uring_batch->ring,
                                   io_uring_batch->eventfd_);
    if (rc < 0) {
        ::close(io_uring_batch->eventfd_);
        io_uring_batch->eventfd_ = -1;
        io_uring_queue_exit(&io_uring_batch->ring);
        Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
        batch = nullptr;
        return Status::InternalError(
            std::string("io_uring_register_eventfd failed: ") + strerror(-rc) +
            LOC_MARK);
    }
    rs = reactor_->registerBatch(io_uring_batch);
    if (!rs.ok()) {
        io_uring_unregister_eventfd(&io_uring_batch->ring);
        ::close(io_uring_batch->eventfd_);
        io_uring_batch->eventfd_ = -1;
        io_uring_queue_exit(&io_uring_batch->ring);
        Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
        batch = nullptr;
        return rs;
    }
    {
        std::lock_guard<std::mutex> lock(allocated_batches_mutex_);
        allocated_batches_.insert(io_uring_batch);
    }
    return Status::OK();
}

Status IOUringTransport::freeSubBatch(SubBatchRef& batch) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (!io_uring_batch)
        return Status::InvalidArgument("Invalid IO Uring sub-batch" LOC_MARK);

    // 1. Detach from the reactor and wait out any worker task (covers its
    //    notifyTerminal). On timeout, leak the batch rather than free it.
    if (reactor_ && !reactor_->unregisterBatch(io_uring_batch)) {
        return Status::InternalError(
            "io_uring batch still has an active reactor worker" LOC_MARK);
    }

    // 2. Retire in-flight IO. io_uring_queue_exit does not cancel in-flight
    //    requests, and the kernel DMAs into the bounce buffers owned by
    //    task_list -- they must not be freed until every CQE has landed.
    //    The reactor no longer watches this batch, so reap here ourselves.
    //    io_uring_submit flushes any SQEs a failed submit left queued, so
    //    their CQEs (already counted in pending_cqes) can arrive.
    constexpr auto kReapTimeout = std::chrono::seconds(5);
    const auto deadline = std::chrono::steady_clock::now() + kReapTimeout;
    std::vector<HarvestedCqe> done;
    while (io_uring_batch->pending_cqes.load(std::memory_order_acquire) > 0) {
        done.clear();
        {
            std::lock_guard<std::mutex> lk(io_uring_batch->ring_mutex);
            (void)io_uring_submit(&io_uring_batch->ring);
            harvestCompletionsLocked(io_uring_batch, done);
        }
        for (const auto& h : done) finalizeCompletion(io_uring_batch, h);
        if (io_uring_batch->pending_cqes.load(std::memory_order_acquire) == 0)
            break;
        if (std::chrono::steady_clock::now() >= deadline) {
            return Status::InternalError(
                "io_uring batch still has in-flight IO" LOC_MARK);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    {
        std::lock_guard<std::mutex> lock(allocated_batches_mutex_);
        allocated_batches_.erase(io_uring_batch);
    }
    if (io_uring_batch->eventfd_ >= 0) {
        io_uring_unregister_eventfd(&io_uring_batch->ring);
        ::close(io_uring_batch->eventfd_);
        io_uring_batch->eventfd_ = -1;
    }
    io_uring_queue_exit(&io_uring_batch->ring);
    Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
    batch = nullptr;
    return Status::OK();
}

std::string IOUringTransport::getIOUringFilePath(SegmentID target_id) {
    std::string ret;
    auto status = metadata_->segmentManager().withCachedSegment(
        target_id, [&](SegmentDesc* segment) {
            if (segment->type != SegmentType::File)
                return Status::NeedsRefreshCache(
                    "Segment type is not File" LOC_MARK);
            auto& detail = std::get<FileSegmentDesc>(segment->detail);
            if (detail.buffers.empty())
                return Status::NeedsRefreshCache("No buffers found" LOC_MARK);
            ret = detail.buffers[0].path;
            return Status::OK();
        });
    if (!status.ok()) return "";
    return ret;
}

IOUringFileContext* IOUringTransport::findFileContext(SegmentID target_id) {
    thread_local FileContextMap tl_file_context_map;
    if (tl_file_context_map.count(target_id))
        return tl_file_context_map[target_id].get();

    RWSpinlock::WriteGuard guard(file_context_lock_);
    if (!file_context_map_.count(target_id)) {
        std::string path = getIOUringFilePath(target_id);
        if (path.empty()) return nullptr;
        file_context_map_[target_id] =
            std::make_shared<IOUringFileContext>(path);
    }

    tl_file_context_map = file_context_map_;
    return tl_file_context_map[target_id].get();
}

Status IOUringTransport::submitTransferTasks(
    SubBatchRef batch, const std::vector<Request>& request_list) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (!io_uring_batch)
        return Status::InvalidArgument("Invalid IO Uring sub-batch" LOC_MARK);

    // Phase 1: validate and stage everything private (file contexts, bounce
    // buffers, WRITE copy-in) before taking ring_mutex, so the lock never
    // covers a large memcpy and a failure here leaves no shared state to
    // unwind.
    struct StagedRequest {
        const Request* req;
        IOUringFileContext* context;
        IOUringTask::OwnedBuffer bounce{nullptr, &std::free};
    };
    std::vector<StagedRequest> staged;
    staged.reserve(request_list.size());
    for (auto& request : request_list) {
        if (request.opcode != Request::READ && request.opcode != Request::WRITE)
            return Status::InvalidArgument("Unsupported opcode" LOC_MARK);
        IOUringFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready())
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);
        StagedRequest s{&request, context};
        const size_t kPageSize = 4096;
        if (Platform::getLoader().getMemoryType(request.source) == MTYPE_CUDA ||
            (uint64_t)request.source % kPageSize) {
            void* aligned_buffer = nullptr;
            int rc = posix_memalign(&aligned_buffer, kPageSize, request.length);
            if (rc)
                return Status::InternalError("posix_memalign failed" LOC_MARK);
            s.bounce.reset(aligned_buffer);
            if (request.opcode == Request::WRITE)
                Platform::getLoader().copy(s.bounce.get(), request.source,
                                           request.length);
        }
        staged.push_back(std::move(s));
    }

    // Phase 2: publish into the ring. Everything under the lock is
    // microsecond-scale; it serializes against the reactor worker draining
    // CQEs on the same ring.
    std::lock_guard<std::mutex> ring_lk(io_uring_batch->ring_mutex);
    if (staged.size() + io_uring_batch->task_list.size() >
        io_uring_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);
    const size_t first_new_task = io_uring_batch->task_list.size();
    const unsigned saved_sqe_tail = io_uring_batch->ring.sq.sqe_tail;
    for (auto& s : staged) {
        io_uring_batch->task_list.emplace_back();
        auto& task = io_uring_batch->task_list.back();
        task.request = *s.req;
        task.buffer = std::move(s.bounce);

        auto sqe = io_uring_get_sqe(&io_uring_batch->ring);
        if (!sqe) {
            // Discard prepped-but-unflushed SQEs and their tasks; nothing has
            // reached the kernel, so no user_data is live.
            io_uring_batch->ring.sq.sqe_tail = saved_sqe_tail;
            io_uring_batch->task_list.resize(first_new_task);
            return Status::InternalError("io_uring SQ full" LOC_MARK);
        }
        void* buf = task.buffer ? task.buffer.get() : task.request.source;
        if (task.request.opcode == Request::READ)
            io_uring_prep_read(sqe, s.context->getHandle(), buf,
                               task.request.length, task.request.target_offset);
        else
            io_uring_prep_write(sqe, s.context->getHandle(), buf,
                                task.request.length,
                                task.request.target_offset);
        sqe->user_data = (uintptr_t)&task;
    }

    // Count in-flight CQEs before submit so a completion racing the submitter
    // still observes a non-zero counter. Never decrement on failure: once
    // io_uring_submit is called the prepped SQEs are flushed into the
    // kernel-visible SQ ring and WILL each produce a CQE once a later submit
    // (or freeSubBatch's reap loop) hands them to the kernel.
    io_uring_batch->pending_cqes.fetch_add(staged.size(),
                                           std::memory_order_acq_rel);
    int rc = io_uring_submit(&io_uring_batch->ring);
    if (rc < 0) {
        return Status::InternalError(std::string("io_uring_submit failed: ") +
                                     strerror(-rc) + LOC_MARK);
    }
    if ((size_t)rc < staged.size()) {
        LOG(WARNING) << "io_uring_submit consumed " << rc << "/"
                     << staged.size()
                     << " SQEs; remainder stays queued";
        return Status::InternalError("partial io_uring_submit" LOC_MARK);
    }
    return Status::OK();
}

void IOUringTransport::harvestCompletionsLocked(
    IOUringSubBatch* batch, std::vector<HarvestedCqe>& out) {
    struct io_uring_cqe* cqe = nullptr;
    while (io_uring_peek_cqe(&batch->ring, &cqe) == 0 && cqe) {
        auto* task = reinterpret_cast<IOUringTask*>(cqe->user_data);
        if (task) out.push_back({task, cqe->res});
        io_uring_cqe_seen(&batch->ring, cqe);
        cqe = nullptr;
    }
}

bool IOUringTransport::finalizeCompletion(IOUringSubBatch* batch,
                                          const HarvestedCqe& h) {
    auto* task = h.task;
    TransferStatusEnum final_status = TransferStatusEnum::COMPLETED;
    if (h.res < 0) {
        LOG(INFO) << "Received an event with error code " << h.res;
        final_status = TransferStatusEnum::FAILED;
    } else {
        if (task->buffer) {
            if (task->request.opcode == Request::READ)
                Platform::getLoader().copy(task->request.source,
                                           task->buffer.get(),
                                           task->request.length);
            task->buffer.reset();
        }
        task->transferred_bytes.store(task->request.length,
                                      std::memory_order_release);
    }

    auto expected = TransferStatusEnum::PENDING;
    if (task->status_word.compare_exchange_strong(
            expected, final_status, std::memory_order_acq_rel)) {
        batch->pending_cqes.fetch_sub(1, std::memory_order_acq_rel);
        return true;
    }
    return false;
}

Status IOUringTransport::getTransferStatus(SubBatchRef batch, int task_id,
                                           TransferStatus& status) {
    auto io_uring_batch = dynamic_cast<IOUringSubBatch*>(batch);
    if (!io_uring_batch)
        return Status::InvalidArgument("Invalid IO Uring sub-batch" LOC_MARK);
    if (task_id < 0 || task_id >= (int)io_uring_batch->task_list.size())
        return Status::InvalidArgument("Invalid task ID");
    auto& task = io_uring_batch->task_list[task_id];
    status = TransferStatus{
        task.status_word.load(std::memory_order_acquire),
        task.transferred_bytes.load(std::memory_order_acquire)};
    // Fallback reap for batches without a sink: such callers have no event
    // path and rely on polling for completions to be processed. The reactor
    // also drains this ring; ring_mutex and the PENDING->terminal CAS make
    // the two reapers safe to coexist.
    if (task.status_word.load(std::memory_order_acquire) ==
            TransferStatusEnum::PENDING &&
        !io_uring_batch->sink) {
        std::vector<HarvestedCqe> done;
        {
            std::lock_guard<std::mutex> lock(io_uring_batch->ring_mutex);
            harvestCompletionsLocked(io_uring_batch, done);
        }
        for (const auto& h : done) finalizeCompletion(io_uring_batch, h);
        status = TransferStatus{
            task.status_word.load(std::memory_order_acquire),
            task.transferred_bytes.load(std::memory_order_acquire)};
    }
    return Status::OK();
}

Status IOUringTransport::addMemoryBuffer(BufferDesc& desc,
                                         const MemoryOptions& options) {
    return Status::OK();
}

Status IOUringTransport::removeMemoryBuffer(BufferDesc& desc) {
    return Status::OK();
}

}  // namespace tent
}  // namespace mooncake
