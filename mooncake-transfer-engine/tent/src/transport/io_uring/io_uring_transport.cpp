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

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int default_workers = std::max(2, static_cast<int>(hw / 4));
    const int workers =
        conf_ ? conf_->get<int>("transports/io_uring/reactor_workers",
                                default_workers)
              : default_workers;
    reactor_ = std::make_unique<IOUringReactor>();
    auto rs = reactor_->start(static_cast<size_t>(workers));
    if (!rs.ok()) {
        reactor_.reset();
        return rs;
    }

    installed_ = true;
    async_memcpy_threshold_ =
        conf_->get("transports/nvlink/async_memcpy_threshold", 1024) * 1024;
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

Status IOUringTransport::allocateSubBatch(SubBatchRef& batch, size_t max_size) {
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
    if (reactor_) {
        auto rs = reactor_->registerBatch(io_uring_batch);
        if (!rs.ok()) {
            io_uring_unregister_eventfd(&io_uring_batch->ring);
            ::close(io_uring_batch->eventfd_);
            io_uring_batch->eventfd_ = -1;
            io_uring_queue_exit(&io_uring_batch->ring);
            Slab<IOUringSubBatch>::Get().deallocate(io_uring_batch);
            batch = nullptr;
            return rs;
        }
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
    {
        std::lock_guard<std::mutex> lock(allocated_batches_mutex_);
        allocated_batches_.erase(io_uring_batch);
    }
    if (reactor_) reactor_->unregisterBatch(io_uring_batch);
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
    if (request_list.size() + (int)io_uring_batch->task_list.size() >
        io_uring_batch->max_size)
        return Status::TooManyRequests("Exceed batch capacity" LOC_MARK);
    // Hold ring_mutex across SQE prep + submit so we serialize against the
    // reactor-dispatched worker that drains CQEs on the same ring.
    std::lock_guard<std::mutex> ring_lk(io_uring_batch->ring_mutex);
    for (auto& request : request_list) {
        io_uring_batch->task_list.emplace_back();
        auto& task = io_uring_batch->task_list.back();
        task.request = request;
        task.status_word.store(TransferStatusEnum::PENDING,
                               std::memory_order_release);
        task.transferred_bytes.store(0, std::memory_order_release);

        IOUringFileContext* context = findFileContext(request.target_id);
        if (!context || !context->ready())
            return Status::InvalidArgument("Invalid remote segment" LOC_MARK);

        struct io_uring_sqe* sqe = io_uring_get_sqe(&io_uring_batch->ring);
        if (!sqe)
            return Status::InternalError("io_uring_get_sqe failed" LOC_MARK);

        const size_t kPageSize = 4096;
        if (Platform::getLoader().getMemoryType(request.source) == MTYPE_CUDA ||
            (uint64_t)request.source % kPageSize) {
            void* aligned_buffer = nullptr;
            int rc = posix_memalign(&aligned_buffer, kPageSize, request.length);
            if (rc)
                return Status::InternalError("posix_memalign failed" LOC_MARK);
            task.buffer.reset(aligned_buffer);

            if (request.opcode == Request::READ)
                io_uring_prep_read(sqe, context->getHandle(),
                                   task.buffer.get(),
                                   request.length, request.target_offset);
            else if (request.opcode == Request::WRITE) {
                Platform::getLoader().copy(task.buffer.get(), request.source,
                                           request.length);
                io_uring_prep_write(sqe, context->getHandle(),
                                    task.buffer.get(),
                                    request.length, request.target_offset);
            }
        } else {
            if (request.opcode == Request::READ)
                io_uring_prep_read(sqe, context->getHandle(), request.source,
                                   request.length, request.target_offset);
            else if (request.opcode == Request::WRITE)
                io_uring_prep_write(sqe, context->getHandle(), request.source,
                                    request.length, request.target_offset);
        }
        sqe->user_data = (uintptr_t)&task;
    }

    // Account for in-flight CQEs before submit so a fast completion racing the
    // submitter still observes a non-zero counter.
    io_uring_batch->pending_cqes.fetch_add(request_list.size(),
                                           std::memory_order_acq_rel);
    int rc = io_uring_submit(&io_uring_batch->ring);
    if (rc != (int32_t)request_list.size()) {
        // Roll the counter back; the failed SQEs will never produce CQEs.
        io_uring_batch->pending_cqes.fetch_sub(request_list.size(),
                                               std::memory_order_acq_rel);
        return Status::InternalError(std::string("io_uring_submit failed: ") +
                                     strerror(-rc) + LOC_MARK);
    }
    return Status::OK();
}

bool IOUringTransport::processCompletionStatic(IOUringSubBatch* batch,
                                                struct io_uring_cqe* cqe) {
    if (!batch || !cqe) return false;
    auto* task = reinterpret_cast<IOUringTask*>(cqe->user_data);
    if (!task) return false;

    TransferStatusEnum final_status = TransferStatusEnum::COMPLETED;
    if (cqe->res < 0) {
        LOG(INFO) << "Received an event with error code " << cqe->res;
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
    if (task_id < 0 || task_id >= (int)io_uring_batch->task_list.size())
        return Status::InvalidArgument("Invalid task ID");
    auto& task = io_uring_batch->task_list[task_id];
    status = TransferStatus{
        task.status_word.load(std::memory_order_acquire),
        task.transferred_bytes.load(std::memory_order_acquire)};
    // Fallback peek path: only when the user hasn't installed a sink. With a
    // sink, the reactor + worker pool drives completion and we must not race
    // against them on the same ring.
    if (task.status_word.load(std::memory_order_acquire) ==
            TransferStatusEnum::PENDING &&
        !io_uring_batch->sink) {
        struct io_uring_cqe* cqe = nullptr;
        std::lock_guard<std::mutex> lock(io_uring_batch->ring_mutex);
        bool any_terminal = false;
        while (true) {
            cqe = nullptr;
            int err = io_uring_peek_cqe(&io_uring_batch->ring, &cqe);
            if (err == -EAGAIN || !cqe) break;
            if (err) {
                return Status::InternalError(
                    std::string("io_uring_peek_cqe failed: ") + strerror(-err));
            }
            if (processCompletionStatic(io_uring_batch, cqe)) any_terminal = true;
            io_uring_cqe_seen(&io_uring_batch->ring, cqe);
        }
        (void)any_terminal;  // no sink to notify
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
