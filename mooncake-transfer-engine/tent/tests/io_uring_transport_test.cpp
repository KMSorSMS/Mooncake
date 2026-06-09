#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tent/common/config.h"
#include "tent/common/types.h"
#include "tent/runtime/control_plane.h"
#include "tent/runtime/segment.h"
#include "tent/runtime/topology.h"
#include "tent/runtime/transport.h"
#include "tent/transport/io_uring/io_uring_transport.h"

namespace mooncake {
namespace tent {
namespace {

constexpr size_t kFileSize = 8192;
constexpr size_t kTransferSize = 4096;
std::atomic<uint64_t> g_remote_handle_salt{0};

class CountingSink final : public BatchEventSink {
   public:
    void notifyMaybeReady() noexcept override {
        notify_count_.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard<std::mutex> lock(mu_);
        cv_.notify_all();
    }

    void close() noexcept override { closed_.store(true, std::memory_order_release); }

    int notifyCount() const { return notify_count_.load(std::memory_order_acquire); }

    bool waitForNotify(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] {
            return notify_count_.load(std::memory_order_acquire) > 0;
        });
    }

   private:
    std::atomic<int> notify_count_{0};
    std::atomic<bool> closed_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

class IOUringTransportTest : public ::testing::Test {
   protected:
    void SetUp() override {
        char templ[] = "/tmp/tent_iouring_transport_testXXXXXX";
        const int fd = mkstemp(templ);
        ASSERT_GE(fd, 0) << std::strerror(errno);
        temp_path_ = templ;
        ASSERT_EQ(ftruncate(fd, static_cast<off_t>(kFileSize)), 0)
            << std::strerror(errno);
        ASSERT_EQ(close(fd), 0) << std::strerror(errno);

        conf_ = std::make_shared<Config>();
        metadata_ = std::make_shared<ControlService>("p2p", "", nullptr);
        topology_ = std::make_shared<Topology>();

        auto local_desc = metadata_->segmentManager().getLocal();
        local_desc->machine_id = "test-machine";
        local_desc->rpc_server_addr = "127.0.0.1:0";

        // findFileContext() keeps a thread-local cache keyed by SegmentID.
        // Reusing LOCAL_SEGMENT_ID across tests can accidentally reuse a file
        // context from a previous fixture, so consume a unique remote handle
        // per test case.
        const auto salt =
            g_remote_handle_salt.fetch_add(1, std::memory_order_relaxed);
        for (uint64_t i = 0; i < salt; ++i) {
            SegmentID ignored = 0;
            ASSERT_TRUE(metadata_->segmentManager()
                            .openRemote(ignored, kLocalFileSegmentPrefix +
                                                     temp_path_.string() +
                                                     ".unused." +
                                                     std::to_string(i))
                            .ok());
        }
        ASSERT_TRUE(metadata_->segmentManager()
                        .openRemote(target_segment_id_,
                                    kLocalFileSegmentPrefix + temp_path_.string())
                        .ok());

        auto status =
            transport_.install(local_segment_name_, metadata_, topology_, conf_);
        if (!status.ok()) {
            GTEST_SKIP() << "io_uring unavailable: " << status.ToString();
        }
        installed_ = true;
    }

    void TearDown() override {
        if (active_batch_) {
            EXPECT_TRUE(transport_.freeSubBatch(active_batch_).ok());
        }
        if (installed_) {
            EXPECT_TRUE(transport_.uninstall().ok());
        }
        if (!temp_path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(temp_path_, ec);
        }
    }

    void AllocateBatch(size_t capacity = 1) {
        ASSERT_TRUE(transport_.allocateSubBatch(active_batch_, capacity).ok());
        ASSERT_NE(active_batch_, nullptr);
        io_batch_ = dynamic_cast<IOUringSubBatch*>(active_batch_);
        ASSERT_NE(io_batch_, nullptr);
    }

    static std::vector<uint8_t> makePattern(size_t size, uint8_t seed) {
        std::vector<uint8_t> bytes(size);
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<uint8_t>(seed + (i % 31));
        }
        return bytes;
    }

    void writeFile(const std::vector<uint8_t>& bytes) {
        const int fd = open(temp_path_.c_str(), O_WRONLY);
        ASSERT_GE(fd, 0) << std::strerror(errno);
        const ssize_t written =
            pwrite(fd, bytes.data(), bytes.size(), /*offset=*/0);
        ASSERT_EQ(written, static_cast<ssize_t>(bytes.size()))
            << std::strerror(errno);
        ASSERT_EQ(close(fd), 0) << std::strerror(errno);
    }

    std::vector<uint8_t> readFile(size_t size) {
        std::vector<uint8_t> bytes(size);
        const int fd = open(temp_path_.c_str(), O_RDONLY);
        EXPECT_GE(fd, 0) << std::strerror(errno);
        if (fd < 0) return {};
        const ssize_t nread = pread(fd, bytes.data(), size, /*offset=*/0);
        EXPECT_EQ(nread, static_cast<ssize_t>(size)) << std::strerror(errno);
        EXPECT_EQ(close(fd), 0) << std::strerror(errno);
        return bytes;
    }

    bool waitUntilCompleted(TransferStatus& status,
                            std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            status = {};
            if (!transport_.getTransferStatus(active_batch_, 0, status).ok()) {
                return false;
            }
            if (status.s != TransferStatusEnum::PENDING) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    }

    std::shared_ptr<Config> conf_;
    std::shared_ptr<ControlService> metadata_;
    std::shared_ptr<Topology> topology_;
    IOUringTransport transport_;
    std::filesystem::path temp_path_;
    std::string local_segment_name_ = "io-uring-test-segment";
    SegmentID target_segment_id_ = LOCAL_SEGMENT_ID;
    Transport::SubBatchRef active_batch_ = nullptr;
    IOUringSubBatch* io_batch_ = nullptr;
    bool installed_ = false;
};

TEST_F(IOUringTransportTest, EventDrivenSubmitNotifiesAndDrainCompletes) {
    AllocateBatch();

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    auto storage = makePattern(kTransferSize + 1, 0x41);
    auto expected =
        std::vector<uint8_t>(storage.begin() + 1, storage.begin() + 1 + kTransferSize);

    Request request{};
    request.opcode = Request::WRITE;
    request.source = storage.data() + 1;
    request.target_id = target_segment_id_;
    request.target_offset = 0;
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());
    EXPECT_TRUE(io_batch_->registered.load(std::memory_order_acquire));

    ASSERT_TRUE(sink->waitForNotify(std::chrono::milliseconds(2000)));

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, kTransferSize);
    EXPECT_GE(sink->notifyCount(), 1);
    EXPECT_EQ(readFile(kTransferSize), expected);

    ASSERT_TRUE(transport_.drain().ok());
    EXPECT_EQ(io_batch_->pending_cqes.load(std::memory_order_acquire), 0u);
}

TEST_F(IOUringTransportTest, GetTransferStatusFallsBackWithoutSink) {
    AllocateBatch();

    auto storage = makePattern(kTransferSize + 1, 0x11);
    auto expected =
        std::vector<uint8_t>(storage.begin() + 1, storage.begin() + 1 + kTransferSize);

    Request request{};
    request.opcode = Request::WRITE;
    request.source = storage.data() + 1;
    request.target_id = target_segment_id_;
    request.target_offset = 0;
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, kTransferSize);
    EXPECT_EQ(readFile(kTransferSize), expected);
}

TEST_F(IOUringTransportTest, ReadOpcodeRoundTrip) {
    AllocateBatch();

    auto on_disk = makePattern(kTransferSize, 0x77);
    writeFile(on_disk);

    // Use an aligned destination to exercise the zero-copy path.
    std::vector<uint8_t> dest(kTransferSize, 0);

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    Request request{};
    request.opcode = Request::READ;
    request.source = dest.data();
    request.target_id = target_segment_id_;
    request.target_offset = 0;
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());
    ASSERT_TRUE(sink->waitForNotify(std::chrono::milliseconds(2000)));

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);
    EXPECT_EQ(status.transferred_bytes, kTransferSize);
    EXPECT_EQ(dest, on_disk);
}

TEST_F(IOUringTransportTest, DrainAllowsCleanFreeAndReallocate) {
    AllocateBatch();

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    auto storage = makePattern(kTransferSize + 1, 0x55);

    Request request{};
    request.opcode = Request::WRITE;
    request.source = storage.data() + 1;
    request.target_id = target_segment_id_;
    request.target_offset = 0;
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());
    EXPECT_TRUE(io_batch_->registered.load(std::memory_order_acquire));

    TransferStatus status{};
    ASSERT_TRUE(waitUntilCompleted(status, std::chrono::milliseconds(2000)));
    EXPECT_EQ(status.s, TransferStatusEnum::COMPLETED);

    // Drain waits until in-flight CQEs are reaped by the reactor.
    ASSERT_TRUE(transport_.drain().ok());
    EXPECT_EQ(io_batch_->pending_cqes.load(std::memory_order_acquire), 0u);

    // freeSubBatch must work post-drain without hanging or asserting.
    auto* batch_ref = active_batch_;
    ASSERT_TRUE(transport_.freeSubBatch(batch_ref).ok());
    EXPECT_EQ(batch_ref, nullptr);
    active_batch_ = nullptr;
    io_batch_ = nullptr;

    // Re-allocating after drain must yield a fresh, fully reset batch.
    AllocateBatch();
    EXPECT_TRUE(io_batch_->registered.load(std::memory_order_acquire));
    EXPECT_FALSE(io_batch_->dispatch_pending.load(std::memory_order_acquire));
    EXPECT_EQ(io_batch_->pending_cqes.load(std::memory_order_acquire), 0u);
    EXPECT_TRUE(io_batch_->task_list.empty());
}

TEST_F(IOUringTransportTest, FreeSubBatchWhileWorkerActiveDoesNotHang) {
    AllocateBatch();

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    auto storage = makePattern(kTransferSize + 1, 0x22);

    Request request{};
    request.opcode = Request::WRITE;
    request.source = storage.data() + 1;
    request.target_id = target_segment_id_;
    request.target_offset = 0;
    request.length = kTransferSize;

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, {request}).ok());

    // Free immediately - unregisterBatch barriers must complete promptly.
    auto* batch_ref = active_batch_;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(transport_.freeSubBatch(batch_ref).ok());
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    // 5s barrier timeout + slack.
    EXPECT_LT(elapsed, std::chrono::seconds(10));
    EXPECT_EQ(batch_ref, nullptr);
    active_batch_ = nullptr;
    io_batch_ = nullptr;
}

TEST_F(IOUringTransportTest, ExceedBatchCapacityRejected) {
    AllocateBatch(/*capacity=*/1);

    auto storage = makePattern(kTransferSize + 1, 0x33);

    Request a{};
    a.opcode = Request::WRITE;
    a.source = storage.data() + 1;
    a.target_id = target_segment_id_;
    a.target_offset = 0;
    a.length = kTransferSize;

    Request b = a;

    auto status = transport_.submitTransferTasks(active_batch_, {a, b});
    EXPECT_FALSE(status.ok());
}

TEST_F(IOUringTransportTest, ReactorSurvivesEmptyRingAllocate) {
    // Allocate then immediately free without ever submitting; reactor must
    // tolerate a registered fd that never produces an event.
    AllocateBatch();
    auto* batch_ref = active_batch_;
    ASSERT_TRUE(transport_.freeSubBatch(batch_ref).ok());
    EXPECT_EQ(batch_ref, nullptr);
    active_batch_ = nullptr;
    io_batch_ = nullptr;
}

TEST_F(IOUringTransportTest, ConcurrentSubmitAndCompletionDoesNotRace) {
    constexpr size_t kCapacity = 16;
    constexpr size_t kChunk = 4096;  // O_DIRECT requires block-aligned length
    ASSERT_TRUE(transport_.allocateSubBatch(active_batch_, kCapacity).ok());
    io_batch_ = dynamic_cast<IOUringSubBatch*>(active_batch_);
    ASSERT_NE(io_batch_, nullptr);

    auto sink = std::make_shared<CountingSink>();
    active_batch_->sink = sink;

    // Resize the temp file so all offsets fit.
    const int fd = open(temp_path_.c_str(), O_WRONLY);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, static_cast<off_t>(kCapacity * kChunk)), 0);
    ASSERT_EQ(close(fd), 0);

    std::vector<std::vector<uint8_t>> storages(kCapacity);
    std::vector<Request> requests;
    requests.reserve(kCapacity);
    for (size_t i = 0; i < kCapacity; ++i) {
        // Unaligned source pointer (storage.data()+1) forces the bounce-buffer
        // path through posix_memalign, exercising OwnedBuffer cleanup under
        // contention while keeping the IO length block-aligned for O_DIRECT.
        storages[i] = makePattern(kChunk + 1, static_cast<uint8_t>(0x80 + i));
        Request r{};
        r.opcode = Request::WRITE;
        r.source = storages[i].data() + 1;
        r.target_id = target_segment_id_;
        r.target_offset = static_cast<uint64_t>(i * kChunk);
        r.length = kChunk;
        requests.push_back(r);
    }

    ASSERT_TRUE(transport_.submitTransferTasks(active_batch_, requests).ok());
    ASSERT_TRUE(sink->waitForNotify(std::chrono::milliseconds(2000)));

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(5000);
    bool all_done = false;
    while (std::chrono::steady_clock::now() < deadline) {
        size_t done = 0;
        for (size_t i = 0; i < kCapacity; ++i) {
            TransferStatus s{};
            ASSERT_TRUE(
                transport_.getTransferStatus(active_batch_,
                                             static_cast<int>(i), s)
                    .ok());
            if (s.s == TransferStatusEnum::COMPLETED) ++done;
        }
        if (done == kCapacity) {
            all_done = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_TRUE(all_done);
    EXPECT_EQ(io_batch_->pending_cqes.load(std::memory_order_acquire), 0u);
    EXPECT_GE(sink->notifyCount(), 1);
}

TEST_F(IOUringTransportTest, ReactorDispatchesAcrossManyBatches) {
    constexpr int kBatches = 8;

    std::vector<Transport::SubBatchRef> batches(kBatches, nullptr);
    std::vector<std::shared_ptr<CountingSink>> sinks(kBatches);
    std::vector<std::vector<uint8_t>> storages(kBatches);

    for (int i = 0; i < kBatches; ++i) {
        ASSERT_TRUE(transport_.allocateSubBatch(batches[i], 1).ok());
        sinks[i] = std::make_shared<CountingSink>();
        batches[i]->sink = sinks[i];
        storages[i] = makePattern(kTransferSize + 1,
                                  static_cast<uint8_t>(0xA0 + i));
        Request r{};
        r.opcode = Request::WRITE;
        r.source = storages[i].data() + 1;  // unaligned -> bounce buffer
        r.target_id = target_segment_id_;
        r.target_offset = 0;
        r.length = kTransferSize;
        ASSERT_TRUE(transport_.submitTransferTasks(batches[i], {r}).ok());
    }

    for (int i = 0; i < kBatches; ++i) {
        ASSERT_TRUE(sinks[i]->waitForNotify(std::chrono::milliseconds(5000)))
            << "batch " << i << " never notified";
        TransferStatus s{};
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(2000);
        bool done = false;
        while (std::chrono::steady_clock::now() < deadline) {
            ASSERT_TRUE(
                transport_.getTransferStatus(batches[i], 0, s).ok());
            if (s.s == TransferStatusEnum::COMPLETED) {
                done = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        EXPECT_TRUE(done) << "batch " << i << " not completed";
    }

    for (int i = 0; i < kBatches; ++i) {
        ASSERT_TRUE(transport_.freeSubBatch(batches[i]).ok());
    }
    // Override fixture's TearDown active_batch_ since we managed our own.
    active_batch_ = nullptr;
    io_batch_ = nullptr;
}

}  // namespace
}  // namespace tent
}  // namespace mooncake
