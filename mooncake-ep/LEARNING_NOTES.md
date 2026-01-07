# Mooncake-EP 代码学习笔记

> 本文档记录 mooncake-ep 代码学习过程中的理解和笔记

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [mooncake_backend.h 详解](#2-mooncake_backendh-详解)
3. [Broadcast 完整调用流程](#3-broadcast-完整调用流程)
4. [线程模型与任务调度](#4-线程模型与任务调度)
5. [核心数据结构详解](#5-核心数据结构详解)
6. [MooncakeEpBuffer 与 IBGDA P2P 通信](#6-mooncakeepbuffer-与-ibgda-p2p-通信)

---

## 1. 整体架构概览

### 1.1 目录结构

```
mooncake-ep/
├── include/               # 头文件
│   ├── mooncake_backend.h         # 分布式后端接口
│   ├── mooncake_ep_buffer.h       # EP 缓冲区管理
│   ├── mooncake_ep_event.h        # CUDA 事件封装
│   ├── mooncake_ep_*.cuh          # CUDA 相关配置和工具
│   ├── mooncake_worker.cuh        # Worker 线程接口
│   └── mooncake_ibgda/            # InfiniBand GDA (GPU-Direct RDMA)
│
├── src/                   # 实现文件 (~2300行代码)
│   ├── mooncake_backend.cpp       # 1096行 - 核心后端实现
│   ├── mooncake_ep_buffer.cpp     # 565行 - EP缓冲区管理
│   ├── mooncake_worker_thread.cpp # 225行 - Worker线程逻辑
│   ├── mooncake_ep_kernel.cu      # 770行 - CUDA kernels
│   ├── mooncake_worker.cu         # 359行 - Worker CUDA操作
│   └── mooncake_ibgda/mlx5gda.cpp # 452行 - MLX5 GDA实现
│
├── CMakeLists.txt         # 构建配置
└── setup.py               # Python 扩展构建脚本
```

### 1.2 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| **MooncakeBackend** | mooncake_backend.cpp | PyTorch 分布式后端，继承 `c10d::Backend` |
| **MooncakeEpBuffer** | mooncake_ep_buffer.cpp | Expert Parallelism 的 RDMA 缓冲区管理 |
| **MooncakeWorker** | mooncake_worker_thread.cpp | 异步任务执行的状态机 |
| **CUDA Kernels** | mooncake_ep_kernel.cu | dispatch/combine 的 GPU 实现 |
| **MLX5 GDA** | mlx5gda.cpp | InfiniBand 硬件队列对管理 |

### 1.3 架构层次图

```
PyTorch Distributed API
         ↓
┌─────────────────────────────────────┐
│      MooncakeBackend (c10d)         │  ← 分布式后端接口
├─────────────┬───────────────────────┤
│  CPU Path   │      GPU Path (EP)    │
│  (P2P通信)   │                       │
│     ↓       │          ↓            │
│ Worker      │  MooncakeEpBuffer     │  ← 缓冲区管理
│ Threads     │     ├─ GDR Buffer     │  ← GPU RDMA 内存
│     ↓       │     ├─ IBGDA (QP)     │  ← IB 队列对
│ Transfer    │     └─ NVLink IPC     │  ← 节点内通信
│ Engine      │          ↓            │
│             │   CUDA Kernels        │  ← dispatch/combine
└─────────────┴───────────────────────┘
         ↓
    MLX5 Hardware (InfiniBand)
```

### 1.4 建议阅读顺序

1. **头文件** - 先理解接口设计
   - `mooncake_backend.h` → 了解后端 API
   - `mooncake_ep_buffer.h` → 了解 EP 缓冲区结构
   - `mooncake_worker.cuh` → 了解任务状态机

2. **核心实现**
   - `mooncake_backend.cpp` → 后端主逻辑
   - `mooncake_ep_buffer.cpp` → dispatch/combine 流程

3. **CUDA 层**
   - `mooncake_ep_kernel.cu` → GPU 侧的 dispatch/combine

4. **底层 RDMA**
   - `mlx5gda.cpp` → 硬件队列对操作

---

## 2. mooncake_backend.h 详解

> 文件路径: `include/mooncake_backend.h`

### 2.1 文件概述

这是 Mooncake 分布式后端的核心头文件，定义了 `MooncakeBackend` 类，它是 PyTorch 分布式通信的后端实现。

### 2.2 依赖关系

```cpp
#include <mooncake_worker.cuh>                      // Worker 线程接口
#include <torch/torch.h>                            // PyTorch 核心
#include <torch/csrc/distributed/c10d/Backend.hpp>  // PyTorch 分布式后端基类
#include <transfer_engine.h>                        // Mooncake 传输引擎
#include <queue>                                    // P2P 操作队列
#include <mutex>                                    // 线程同步
#include <condition_variable>                       // 条件变量
#include <atomic>                                   // 原子操作
#include <thread>                                   // 工作线程
```

**关键依赖说明：**

| 依赖 | 说明 |
|------|------|
| `c10d::Backend` | PyTorch 分布式后端的抽象基类，定义了所有集合通信操作的接口 |
| `TransferEngine` | Mooncake 的核心传输引擎，负责底层 RDMA 通信 |
| `MooncakeWorker` | 异步任务管理器，处理 GPU 通信任务 |

### 2.3 类结构详解

#### 2.3.1 MooncakeBackendOptions (配置类)

```cpp
struct MooncakeBackendOptions final : ::c10d::Backend::Options {
    explicit MooncakeBackendOptions(at::Tensor activeRanks)
        : Options{"mooncake"}, activeRanks_{activeRanks} {}

    at::Tensor activeRanks_;   // 当前活跃的 rank 列表
    bool isExtension_ = false; // 是否为扩展模式
};
```

**用途：** 创建 MooncakeBackend 时的配置参数

- `activeRanks_`: 一个 Tensor，包含当前参与通信的所有 rank ID
- `isExtension_`: 标记是否为动态扩展的 group

#### 2.3.2 构造函数

```cpp
MooncakeBackend(
    c10::intrusive_ptr<::c10d::Store> store,  // 分布式 KV 存储 (用于 rank 间协调)
    int rank,                                   // 当前进程的 rank
    int size,                                   // 总进程数
    c10::intrusive_ptr<MooncakeBackendOptions> options,
    bool isCpu = false                          // 是否为纯 CPU 模式
);
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `store` | PyTorch 分布式 Store，用于 rank 间交换元数据（如地址信息） |
| `rank` | 当前进程在通信组中的编号 (0 ~ size-1) |
| `size` | 通信组的总进程数 |
| `options` | 后端配置选项 |
| `isCpu` | 是否只使用 CPU 缓冲区（不使用 GPU） |

#### 2.3.3 公共接口 - 集合通信操作

所有方法都返回 `c10::intrusive_ptr<c10d::Work>`，这是一个异步操作句柄，可用于等待操作完成。

##### 点对点通信 (P2P)

```cpp
// 发送 tensor 到目标 rank
c10::intrusive_ptr<c10d::Work> send(
    std::vector<at::Tensor>& tensors,  // 要发送的 tensor（只支持单个）
    int dstRank,                        // 目标 rank
    int tag                             // 消息标签
) override;

// 从源 rank 接收 tensor
c10::intrusive_ptr<c10d::Work> recv(
    std::vector<at::Tensor>& tensors,  // 接收缓冲区
    int srcRank,                        // 源 rank
    int tag                             // 消息标签
) override;
```

##### 集合通信操作

```cpp
// 广播: 将 root rank 的数据广播给所有 rank
c10::intrusive_ptr<c10d::Work> broadcast(
    std::vector<at::Tensor>& tensors,
    const c10d::BroadcastOptions& opts
) override;

// AllReduce: 所有 rank 的数据规约后，结果分发给所有 rank
c10::intrusive_ptr<c10d::Work> allreduce(
    std::vector<at::Tensor>& tensors,
    const c10d::AllreduceOptions& opts
) override;

// AllGather: 收集所有 rank 的数据到每个 rank
c10::intrusive_ptr<c10d::Work> allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts
) override;

// ReduceScatter: 规约后分散到各 rank
c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const c10d::ReduceScatterOptions& opts
) override;

// AllToAll: 全交换 - 每个 rank 向每个 rank 发送不同的数据
c10::intrusive_ptr<c10d::Work> alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllToAllOptions& opts
) override;

// Barrier: 同步屏障 - 等待所有 rank 到达
c10::intrusive_ptr<c10d::Work> barrier(
    const c10d::BarrierOptions& opts
) override;
```

##### 管理接口

```cpp
void shutdown() override;                              // 关闭后端
static void setHostIp(const std::string& hostIp);      // 设置主机 IP
static void setDeviceFilter(std::vector<std::string>); // 设置网卡过滤器
std::string getPreferredHca(std::string location);     // 获取最优 HCA 设备
at::Tensor getActiveRanksTensor();                     // 获取活跃 rank 列表
int getNumSyncedRanks();                               // 获取已同步的 rank 数
void extendGroupSizeTo(int size);                      // 动态扩展组大小
std::vector<bool> getPeerState(const std::vector<int>& ranks); // 获取 peer 状态
void recoverRanks(const std::vector<int>& ranks);      // 恢复失败的 rank
```

### 2.4 私有成员详解

#### 2.4.1 P2P 操作结构体

```cpp
enum class P2POpType { SEND, RECV };

struct P2POp {
    P2POpType opType;           // 操作类型
    at::Tensor tensor;          // 连续内存的 tensor
    at::Tensor originalTensor;  // 原始 tensor (可能非连续)
    int peerRank;               // 对端 rank
    int tag;                    // 消息标签
    int64_t seq;                // 序列号 (用于排序)
    std::shared_ptr<std::atomic<bool>> completed;  // 完成标志
    std::shared_ptr<std::string> errorMsg;         // 错误信息
};
```

**设计说明：**
- `tensor` vs `originalTensor`: PyTorch tensor 可能不是连续内存布局，需要先 `contiguous()` 再传输
- `seq`: 保证操作顺序，避免乱序问题
- `completed`: 异步完成通知

#### 2.4.2 静态成员 (全局共享)

```cpp
static TransferEngine engine_;      // 传输引擎 (单例)
static bool engineInitialized_;     // 引擎是否已初始化
static int backendIndex_;           // 后端实例索引
static std::string hostIp_;         // 本机 IP 地址
static MooncakeWorker worker_;      // Worker 管理器 (单例)
```

**为什么是 static？**
- 多个 process group 共享同一个 TransferEngine
- 避免重复初始化 RDMA 资源

#### 2.4.3 缓冲区指针

```cpp
void* send_buffer_[2];           // 发送缓冲区 (双缓冲)
void* recv_buffer_[2];           // 接收缓冲区 (双缓冲)
int32_t* cpu_sync_send_region_[2]; // CPU 同步发送区域
int32_t* cpu_sync_recv_region_[2]; // CPU 同步接收区域
int32_t* warmup_send_region_;    // 预热发送区域
int32_t* warmup_recv_region_;    // 预热接收区域
```

**双缓冲设计：**
- `buffer_[0]` 和 `buffer_[1]` 交替使用
- 当一个缓冲区在传输时，另一个可以准备数据
- 实现流水线，隐藏通信延迟

#### 2.4.4 P2P 异步基础设施

```cpp
// 发送队列及其同步原语
std::queue<P2POp> p2pSendQueue_;
std::mutex p2pSendQueueMutex_;
std::condition_variable p2pSendQueueCv_;
std::atomic<bool> p2pSendWorkerRunning_{false};
std::thread p2pSendWorkerThread_;

// 接收队列及其同步原语
std::queue<P2POp> p2pRecvQueue_;
std::mutex p2pRecvQueueMutex_;
std::condition_variable p2pRecvQueueCv_;
std::atomic<bool> p2pRecvWorkerRunning_{false};
std::thread p2pRecvWorkerThread_;
```

**生产者-消费者模型：**

```
主线程                     发送 Worker 线程
   │                            │
   ├─ send() ──────────────────→│
   │   ├─ 创建 P2POp             │
   │   ├─ 加锁                   │
   │   ├─ 入队 p2pSendQueue_     │
   │   ├─ notify_one()          │
   │   └─ 返回 Work 句柄         │
   │                            ├─ wait() 被唤醒
   │                            ├─ 出队
   │                            ├─ processSendOp()
   │                            │   └─ TransferEngine 发送
   │                            └─ 标记 completed
```

#### 2.4.5 其他成员

```cpp
TransferGroupMeta meta_;          // 通信组元数据
bool isShutdown_{false};          // 是否已关闭
int nextRankForConnection_ = 0;   // 下一个待连接的 rank
bool isCpu_{false};               // 是否为 CPU 模式
```

### 2.5 关键设计模式总结

1. **继承 PyTorch Backend**: 通过继承 `c10d::Backend`，无缝集成到 PyTorch 分布式框架

2. **静态单例**: TransferEngine 和 MooncakeWorker 作为静态成员，多个 backend 实例共享

3. **双缓冲**: 发送/接收缓冲区使用双缓冲，实现通信与计算重叠

4. **异步队列**: P2P 操作通过生产者-消费者队列实现真正的异步

5. **故障恢复**: 提供 `getPeerState()` 和 `recoverRanks()` 支持弹性训练

### 2.6 与其他组件的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    MooncakeBackend                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ P2P Send Queue  │  │ P2P Recv Queue  │  ← 异步操作队列   │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────┐                   │
│  │         TransferEngine (静态)        │ ← RDMA 传输引擎   │
│  └─────────────────────────────────────┘                   │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────┐                   │
│  │        MooncakeWorker (静态)         │ ← GPU 任务管理    │
│  └─────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  InfiniBand Hardware   │
          └────────────────────────┘
```

---

## 3. Broadcast 完整调用流程

> 本节以 broadcast 操作为例，详细分析从用户调用到 RDMA 传输的完整流程。

### 3.1 场景假设

- 3 个 rank：0, 1, 2
- **root = 0**（rank 0 广播数据给 rank 1, 2）

### 3.2 调用链概览

```
Python: dist.broadcast(tensor, src=0)
    ↓
MooncakeBackend::broadcast()        # mooncake_backend.cpp
    ↓
MooncakeWorker::putTaskCpu()        # mooncake_worker.cu
    ↓
Worker 线程轮询                      # mooncake_worker_thread.cpp
    ↓
TransferEngine::submitTransfer()    # RDMA 传输
    ↓
callback() 执行
```

### 3.3 第一步：broadcast 函数入口

> 文件：`mooncake_backend.cpp`

```cpp
c10::intrusive_ptr<c10d::Work> MooncakeBackend::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {

    // 只支持单 tensor（Mooncake 不支持单进程多设备）
    TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);

    auto tensor = tensors.back();
    size_t tensorSize = tensor.numel() * tensor.element_size();
    int64_t root = opts.rootRank + opts.rootTensor;  // 广播源 rank
    bool isRoot = (root == rank_);  // 判断自己是否是 root

    if (isCpu_) {
        return worker_.putTaskCpu(
            c10d::OpType::BROADCAST,
            tensorSize,
            root,
            &meta_,
            // tensorToBuffer: 只有 root 执行
            [=](void* dst, size_t pos, size_t realSize) {
                if (isRoot) {
                    memcpy(dst, (char*)tensor.data_ptr() + pos, realSize);
                }
            },
            // bufferToTensor: 所有 rank 执行
            [=](void* src, size_t pos, size_t realSize) {
                memcpy((char*)tensor.data_ptr() + pos, src, realSize);
            }
        );
    }
    // ... GPU 路径类似，使用 cudaMemcpyAsync
}
```

**关键点：**
- `tensorToBuffer`：只有 root 把数据拷贝到 send_buffer
- `bufferToTensor`：所有 rank 把 recv_buffer 拷贝回 tensor

### 3.4 第二步：putTaskCpu 提交任务

> 文件：`mooncake_worker.cu`

```cpp
c10::intrusive_ptr<c10d::Work> MooncakeWorker::putTaskCpu(...) {
    // 计算分块大小（大 tensor 需要分多次传输）
    size_t chunkSize = ((kBufferSize - 1) / meta->size) & ~(size_t)7;

    // 创建异步 Future
    auto future = c10::make_intrusive<c10::ivalue::Future>(...);

    // 递归处理函数
    auto processNextChunk = std::make_shared<std::function<void()>>();

    *processNextChunk = [...]() {
        // 检查是否全部完成
        if (state->currentPos >= tensorSize) {
            future->markCompleted(c10::IValue());
            return;
        }

        int taskId = cpuTaskCount % 2;        // 双缓冲：任务槽位
        int bufferOffset = meta->bufferBaseIndex + meta->taskCount % 2;  // 双缓冲：缓冲区

        // ① 同步执行：tensor → send_buffer
        tensorToBuffer(send_buffer_addr, pos, realSize);

        // ② 设置 callback（异步执行）
        callbacks_[taskId] = [...]() {
            bufferToTensor(recv_buffer_addr, pos, realSize);
            state->currentPos += realSize;
            (*processNextChunk)();  // 递归处理下一个 chunk
        };

        // ③ 激活任务
        tasks_[taskId].active = true;
        ++meta->taskCount;
    };

    (*processNextChunk)();  // 开始处理
    return c10::make_intrusive<MooncakeWorkCpu>(opType, future);
}
```

**此时各 rank 状态：**

| Rank | send_buffer | task.active |
|------|-------------|-------------|
| 0 (root) | 有数据 ✓ | true |
| 1 | 空 | true |
| 2 | 空 | true |

### 3.5 第三步：Worker 线程处理

> 文件：`mooncake_worker_thread.cpp`

Worker 线程是一个独立的后台线程，不断轮询任务状态：

```cpp
void MooncakeWorker::startWorker() {
    std::thread([this] {
        while (running_) {
            PAUSE();  // CPU 等待指令，省电
            for (size_t i = 0; i < kNumTasks_; ++i) {
                auto &task = tasks_[i];
                if (!task.active) continue;

                // 处理任务...
            }
        }
    }).detach();
}
```

#### 任务状态机

```cpp
enum WorkerTaskStatus {
    IDLE = 0,          // 空闲
    TRANSFERRED_1 = 1, // 数据传输完成
    SIGNALED_1 = 2,    // 信号已发送
    DONE = 3,          // 全部完成
};
```

### 3.6 第四步：判断是否跳过传输

```cpp
// 对于 BROADCAST，只有 root 需要发送数据
bool skipTransfer = (task.opType == c10d::OpType::BROADCAST &&
                     group->rank != task.broadcastRoot);

if (skipTransfer) {
    // rank 1, 2 直接跳到 TRANSFERRED_1 状态
    task_status[i].store(TRANSFERRED_1, ...);
    continue;
}
```

**关键点：只有 root (rank 0) 执行 RDMA 发送！**

### 3.7 第五步：Root 构建并提交 RDMA 请求

```cpp
// 只有 rank 0 执行这段代码
std::vector<TransferRequest> entries;

for (int j = 0; j < group->size; ++j) {  // j = 0, 1, 2
    if (!group->activeRanks[j]) continue;

    uint64_t source = send_buffer[bufferOffset].addr;
    uint64_t target_offset = segmentDescs[j]->buffers[bufferOffset + 2].addr;

    entries.push_back(TransferRequest{
        .opcode = TransferRequest::WRITE,  // RDMA Write 操作
        .source = source,                   // rank 0 的 send_buffer
        .target_id = segmentIDs[j],         // 目标 rank 的内存段
        .target_offset = target_offset,     // 目标 rank 的 recv_buffer
        .length = tensorSize,
    });
}

// 提交 RDMA 传输
task.batchID = group->engine->allocateBatchID(entries.size());
group->engine->submitTransfer(task.batchID, entries);  // ← RDMA 在这里！
```

**RDMA 操作示意：**

```
Rank 0 (root):
  send_buffer ──RDMA Write──→ Rank 0 recv_buffer
  send_buffer ──RDMA Write──→ Rank 1 recv_buffer
  send_buffer ──RDMA Write──→ Rank 2 recv_buffer
```

### 3.8 第六步：等待传输完成 + 信号同步

```cpp
// 状态: TRANSFERRED_1
// 轮询检查 RDMA 是否完成
group->engine->getTransferStatus(task.batchID, task_id, status);
if (status.s == TransferStatusEnum::COMPLETED) {
    // 发送完成信号给所有 rank
    // ...
    task_status[i].store(SIGNALED_1, ...);
}

// 状态: SIGNALED_1
// 等待所有 rank 的信号
for (int j = 0; j < group->size; ++j) {
    if (signal_ptr[j] != 1) {
        all_received = false;
    }
}

if (all_received) {
    task_status[i].store(DONE, ...);
    task.active = false;

    // 调用 callback！
    if (hasCallback_[i]) {
        callbacks_[i]();
    }
}
```

### 3.9 第七步：Callback 执行

```cpp
callbacks_[taskId] = [...]() {
    // 从 recv_buffer 拷贝数据回 tensor
    bufferToTensor(recv_buffer_addr, pos, realSize);
    // 所有 rank 都执行：memcpy(tensor, recv_buffer)

    currentPos += realSize;
    (*processNextChunk)();  // 处理下一个 chunk
};
```

### 3.10 完整流程图

```
         Rank 0 (root)                    Rank 1                     Rank 2
              │                              │                          │
   ┌──────────┴──────────┐        ┌──────────┴──────────┐    ┌──────────┴──────────┐
   │ tensorToBuffer()    │        │ tensorToBuffer()    │    │ tensorToBuffer()    │
   │ tensor→send_buffer ✓│        │ (跳过，isRoot=false)│    │ (跳过，isRoot=false)│
   └──────────┬──────────┘        └──────────┬──────────┘    └──────────┬──────────┘
              │                              │                          │
              │ task.active=true             │ task.active=true         │ task.active=true
              │                              │                          │
   ═══════════╪══════════════════════════════╪══════════════════════════╪═══════════
              │                              │                          │
              ▼ Worker 线程                   ▼ Worker 线程               ▼ Worker 线程
              │                              │                          │
   ┌──────────┴──────────┐        ┌──────────┴──────────┐    ┌──────────┴──────────┐
   │ skipTransfer=false  │        │ skipTransfer=true   │    │ skipTransfer=true   │
   │ 构建 RDMA 请求       │        │ 跳过传输             │    │ 跳过传输             │
   └──────────┬──────────┘        └──────────┬──────────┘    └──────────┬──────────┘
              │                              │                          │
              │ submitTransfer()             │                          │
              │                              │                          │
              ├──── RDMA Write ─────────────→│ recv_buffer 收到数据      │
              ├──── RDMA Write ──────────────────────────────────────→│ recv_buffer 收到数据
              │                              │                          │
   ┌──────────┴──────────┐        ┌──────────┴──────────┐    ┌──────────┴──────────┐
   │ 信号同步             │◄──────►│ 信号同步             │◄──►│ 信号同步             │
   └──────────┬──────────┘        └──────────┬──────────┘    └──────────┬──────────┘
              │                              │                          │
              ▼ callback()                   ▼ callback()               ▼ callback()
   ┌──────────┴──────────┐        ┌──────────┴──────────┐    ┌──────────┴──────────┐
   │ bufferToTensor()    │        │ bufferToTensor()    │    │ bufferToTensor()    │
   │ recv_buffer→tensor  │        │ recv_buffer→tensor  │    │ recv_buffer→tensor  │
   └─────────────────────┘        └─────────────────────┘    └─────────────────────┘
              │                              │                          │
              ▼                              ▼                          ▼
         tensor 有数据 ✓               tensor 有数据 ✓            tensor 有数据 ✓
```

### 3.11 关键设计总结

| 步骤 | 执行者 | 执行位置 | 操作 |
|------|--------|----------|------|
| tensorToBuffer | 只有 root | 主线程（同步） | tensor → send_buffer |
| submitTransfer | 只有 root | Worker 线程 | RDMA Write 到所有 rank |
| 信号同步 | 所有 rank | Worker 线程 | 确保数据到达 |
| bufferToTensor | 所有 rank | Worker 线程（callback） | recv_buffer → tensor |

### 3.12 双缓冲机制

```cpp
int taskId = cpuTaskCount % 2;           // 任务槽位：0 或 1
int bufferOffset = bufferBaseIndex + taskCount % 2;  // 缓冲区：0 或 1
```

缓冲区布局：

```
buffers[0]: 发送缓冲区 A    ─┐
buffers[1]: 发送缓冲区 B     ├─ 交替使用
buffers[2]: 接收缓冲区 A    ─┤
buffers[3]: 接收缓冲区 B    ─┘
```

### 3.13 为什么 PyTorch 接口是 vector<Tensor>？

```cpp
TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
```

PyTorch 设计上支持**单进程多 GPU**（一个 rank 管理多个设备），所以接口是 vector。但 Mooncake 只支持单进程单设备，所以强制要求 size == 1。

---

## 4. 线程模型与任务调度

> 本节详细分析 mooncake-ep 的线程设计、创建方式、任务分发和管理机制。

### 4.1 线程架构概览

Mooncake-EP 使用 **4 种类型的线程**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Mooncake-EP 线程架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────┐    │
│  │   主线程     │    │              后台线程                            │    │
│  │             │    │                                                 │    │
│  │ • 用户调用   │    │  ┌─────────────────┐  ┌─────────────────┐      │    │
│  │ • broadcast │    │  │ Collective      │  │ Connection      │      │    │
│  │ • allreduce │    │  │ Worker Thread   │  │ Poller Thread   │      │    │
│  │ • send/recv │    │  │                 │  │                 │      │    │
│  │ • 提交任务   │    │  │ • 轮询任务状态   │  │ • 建立RDMA连接   │      │    │
│  │ • 立即返回   │    │  │ • 执行RDMA传输   │  │ • 发送预热请求   │      │    │
│  └─────────────┘    │  │ • 调用callback  │  │ • 更新连接状态   │      │    │
│                     │  └─────────────────┘  └─────────────────┘      │    │
│                     │                                                 │    │
│                     │  ┌─────────────────┐  ┌─────────────────┐      │    │
│                     │  │ P2P Send        │  │ P2P Recv        │      │    │
│                     │  │ Worker Thread   │  │ Worker Thread   │      │    │
│                     │  │                 │  │                 │      │    │
│                     │  │ • 条件变量等待   │  │ • 条件变量等待   │      │    │
│                     │  │ • 处理send请求   │  │ • 处理recv请求   │      │    │
│                     │  │ • RDMA发送      │  │ • RDMA接收       │      │    │
│                     │  └─────────────────┘  └─────────────────┘      │    │
│                     └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 线程详细说明

| 线程名称 | 创建位置 | 创建方式 | 等待机制 | 用途 |
|----------|----------|----------|----------|------|
| Collective Worker | `MooncakeWorker` 构造函数 | `detach()` | **轮询 (spin-wait)** | 处理集合通信 |
| Connection Poller | `MooncakeBackend` 构造函数 | `detach()` | sleep 50ms | 建立 RDMA 连接 |
| P2P Send Worker | `startP2PWorker()` | `join()` 可用 | **条件变量** | 处理 send 请求 |
| P2P Recv Worker | `startP2PWorker()` | `join()` 可用 | **条件变量** | 处理 recv 请求 |

### 4.3 线程创建详解

#### 4.3.1 Collective Worker Thread（集合通信工作线程）

##### 创建了多少个线程？

**整个进程只有 1 个 Collective Worker Thread**（不是 4 个！）

原因：`worker_` 是 `MooncakeBackend` 的静态成员，全局唯一。4 是任务槽位数，不是线程数。

##### 创建时机

静态成员在程序启动时（**main 函数执行之前**）自动构造：

```
程序启动
    │
    ▼
静态成员初始化（main 之前）
    │
    ├─→ MooncakeBackend::worker_ 构造
    │       │
    │       ▼
    │   MooncakeWorker::MooncakeWorker()  ← mooncake_worker.cu:232
    │       │
    │       ├─→ 分配 tasks_ 数组（4 个槽位）
    │       │
    │       └─→ startWorker()  ← 创建线程
    │               │
    │               ▼
    │           std::thread(...).detach()  ← mooncake_worker_thread.cpp:16-222
    │
    ▼
main() 开始执行
```

##### 静态成员定义（`mooncake_backend.cpp:27`）

```cpp
// mooncake_backend.cpp:27
MooncakeWorker MooncakeBackend::worker_;  // 静态成员，整个进程只有一个
```

##### 构造函数详解（`mooncake_worker.cu:232-250`）

```cpp
MooncakeWorker::MooncakeWorker() {
    int deviceCount = 0;
    cudaError err = cudaGetDeviceCount(&deviceCount);

    if (!err && deviceCount > 0) {
        // 有 GPU：使用 pinned memory（CPU-GPU 共享内存）
        // cudaHostAllocMapped 让 CPU 和 GPU 可以访问同一块内存
        cudaHostAlloc(&tasks_, kNumTasks_ * sizeof(Task), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&tasks_device_, tasks_, 0);
    } else {
        // 无 GPU：使用普通堆内存
        LOG(WARNING) << "No CUDA device found. Only the `mooncake-cpu` backend "
                        "can be used.";
        tasks_ = new Task[kNumTasks_];
    }

    // 初始化所有槽位为空闲
    for (size_t i = 0; i < kNumTasks_; ++i) {
        tasks_[i].active = false;
    }

    // 启动 worker 线程（只创建 1 个线程！）
    startWorker();
}
```

##### 线程创建详解（`mooncake_worker_thread.cpp:14-222`）

```cpp
void MooncakeWorker::startWorker() {
    running_ = true;

    std::thread([this] {  // ← 创建 1 个新线程
        // 线程局部变量（每个槽位的状态）
        std::atomic<WorkerTaskStatus> task_status[kNumTasks_];  // 4 个状态
        clock::time_point activeTime[kNumTasks_];                // 4 个时间戳

        while (running_) {  // ← 无限循环
            PAUSE();  // CPU 暂停指令，降低功耗，仍是忙等

            // 这 1 个线程轮询 4 个槽位
            for (size_t i = 0; i < kNumTasks_; ++i) {
                auto &task = tasks_[i];
                if (!task.active) {
                    task_status[i].store(IDLE, ...);
                    continue;
                }

                // 状态机处理（见下文）
                // IDLE → TRANSFERRED_1 → SIGNALED_1 → DONE
            }
        }
    }).detach();  // ← 分离线程，无法 join，无法主动停止
}
```

##### 关键信息汇总表

| 项目 | 值 | 代码位置 |
|------|-----|----------|
| **线程数量** | **1 个**（全局唯一） | 静态成员保证 |
| 静态成员定义 | `MooncakeWorker MooncakeBackend::worker_;` | `mooncake_backend.cpp:27` |
| 构造函数 | `MooncakeWorker::MooncakeWorker()` | `mooncake_worker.cu:232-250` |
| 线程创建函数 | `startWorker()` | `mooncake_worker_thread.cpp:14` |
| 线程创建方式 | `std::thread([this]{...}).detach()` | `mooncake_worker_thread.cpp:16,222` |
| **任务槽位数** | **4 个**（`kNumTasks_ = 4`） | `mooncake_worker.cuh:84` |
| 内存类型 | 有 GPU 用 pinned memory，无 GPU 用堆内存 | `mooncake_worker.cu:235-242` |
| 创建时机 | 程序启动时（main 之前） | C++ 静态成员自动构造 |
| 线程生命周期 | `detach()` 后无法主动停止 | `mooncake_worker_thread.cpp:222` |

##### 任务槽位 vs 线程数

```
                    ┌─────────────────────────────────────────┐
                    │     1 个 Collective Worker Thread       │
                    │                                         │
                    │   ┌─────────────────────────────────┐   │
                    │   │       轮询 4 个任务槽位           │   │
                    │   │                                 │   │
                    │   │  tasks_[0] ─┐                   │   │
                    │   │             ├─ CPU 双缓冲        │   │
                    │   │  tasks_[1] ─┘                   │   │
                    │   │                                 │   │
                    │   │  tasks_[2] ─┐                   │   │
                    │   │             ├─ CUDA 双缓冲      │   │
                    │   │  tasks_[3] ─┘                   │   │
                    │   │                                 │   │
                    │   └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
```

##### `kNumTasks_ = 4` 定义位置

```cpp
// mooncake_worker.cuh:84
static constexpr size_t kNumTasks_ = 4;
```

##### 4 个槽位的分配（`mooncake_worker.cu`）

| 槽位索引 | 用于 | 代码行 | 计算方式 |
|----------|------|--------|----------|
| `tasks_[0]` | CPU 任务 A | 行 278 | `cpuTaskCount % 2` → 0 |
| `tasks_[1]` | CPU 任务 B | 行 278 | `cpuTaskCount % 2` → 1 |
| `tasks_[2]` | CUDA 任务 A | 行 336 | `cudaTaskCount % 2 + 2` → 2 |
| `tasks_[3]` | CUDA 任务 B | 行 336 | `cudaTaskCount % 2 + 2` → 3 |

```cpp
// mooncake_worker.cu:278 - CPU 任务
int taskId = cpuTaskCount % 2;        // 结果：0 或 1

// mooncake_worker.cu:336 - CUDA 任务
int taskId = cudaTaskCount % 2 + 2;   // 结果：2 或 3
```

#### 4.3.2 Connection Poller Thread（连接轮询线程）

**创建时机：** 每个 `MooncakeBackend` 实例化时

```cpp
// mooncake_backend.cpp:167
std::thread([this, store, backendIndex] {
    connectionPoller(store, backendIndex);
}).detach();

// mooncake_backend.cpp:687
void MooncakeBackend::connectionPoller(c10::intrusive_ptr<::c10d::Store> store,
                                       int backendIndex) {
    while (!isShutdown_) {
        for (int pollingRank = 0; pollingRank <= nextRankForConnection_; ++pollingRank) {
            if (meta_.peerConnected[pollingRank]) {
                continue;
            }

            // 1. 从 Store 获取对端的 server name
            auto peerServerName = store->get_to_str(serverNameKey);

            // 2. 打开远端内存段
            auto segment_id = engine_.openSegment(peerServerName);
            meta_.segmentIDs[pollingRank] = segment_id;

            // 3. 发送预热请求（建立 RDMA 连接）
            if (backendIndex == 0) {
                engine_.submitTransfer(batchID, warmupRequest);
                // 等待完成...
            }

            meta_.peerConnected[pollingRank] = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
```

**特点：**
- **每个 Backend 一个**：每创建一个 `MooncakeBackend` 就启动一个
- **Sleep 等待**：每 50ms 轮询一次，不是忙等
- **负责连接建立**：打开远端内存段，发送预热请求

#### 4.3.3 P2P Send/Recv Worker Threads（点对点通信工作线程）

**创建时机：** `MooncakeBackend` 构造函数末尾

```cpp
// mooncake_backend.cpp:762
void MooncakeBackend::startP2PWorker() {
    p2pSendWorkerRunning_ = true;
    p2pRecvWorkerRunning_ = true;
    p2pSendWorkerThread_ = std::thread(&MooncakeBackend::p2PSendWorkerThread, this);
    p2pRecvWorkerThread_ = std::thread(&MooncakeBackend::p2PRecvWorkerThread, this);
}

// mooncake_backend.cpp:789
void MooncakeBackend::p2PSendWorkerThread() {
    while (p2pSendWorkerRunning_.load()) {
        P2POp op;
        {
            std::unique_lock<std::mutex> lock(p2pSendQueueMutex_);
            // 条件变量等待：有任务或者要停止时唤醒
            p2pSendQueueCv_.wait(lock, [this] {
                return !p2pSendQueue_.empty() || !p2pSendWorkerRunning_.load();
            });

            if (!p2pSendWorkerRunning_.load() && p2pSendQueue_.empty()) {
                break;
            }

            op = p2pSendQueue_.front();
            p2pSendQueue_.pop();
        }

        // 处理发送操作
        processSendOp(op);
        op.completed->store(true, std::memory_order_release);
    }
}
```

**特点：**
- **条件变量等待**：不是忙等，CPU 可以休息
- **可以 join**：析构时会调用 `stopP2PWorker()` 等待线程结束
- **生产者-消费者模型**：主线程生产，Worker 线程消费

### 4.4 任务提交与分发机制

#### 4.4.1 集合通信任务流程

```
主线程                              Collective Worker 线程
   │                                        │
   ├─ broadcast()                           │
   │   └─ putTaskCpu()                      │
   │       ├─ tensorToBuffer()  ← 同步执行   │
   │       ├─ 填充 tasks_[taskId]           │
   │       ├─ tasks_[taskId].active = true  │
   │       └─ return (异步)                 │
   │                                        │
   ▼                                        ├─ 轮询检测 task.active
  继续执行                                   ├─ 状态机处理
                                            │   ├─ IDLE → 提交 RDMA
                                            │   ├─ TRANSFERRED_1 → 发信号
                                            │   └─ SIGNALED_1 → 调用 callback
                                            └─ callback()
                                                └─ bufferToTensor()
```

**任务数据结构：**

```cpp
// mooncake_worker.cuh:42
__global__ struct Task {
    volatile bool active = false;      // 任务激活标志
    c10d::OpType opType;               // 操作类型
    size_t tensorSize;                 // 数据大小
    int64_t broadcastRoot;             // 广播 root
    int bufferOffset;                  // 缓冲区偏移
    BatchID batchID;                   // RDMA 批次 ID
    void* transferGroupMeta;           // 通信组元数据
};
```

#### 4.4.2 P2P 通信任务流程

```
主线程                              P2P Send Worker 线程
   │                                        │
   ├─ send()                                │ (条件变量阻塞中...)
   │   ├─ 创建 P2POp                         │
   │   ├─ 加锁                              │
   │   ├─ p2pSendQueue_.push(op)            │
   │   ├─ p2pSendQueueCv_.notify_one() ────→│ (被唤醒)
   │   └─ return Work 句柄                   │
   │                                        ├─ 取出任务
   ▼                                        ├─ processSendOp()
  继续执行                                   │   └─ RDMA 发送
                                            └─ op.completed = true
```

**P2P 操作数据结构：**

```cpp
// mooncake_backend.h (private)
struct P2POp {
    P2POpType opType;           // SEND 或 RECV
    at::Tensor tensor;          // 数据 tensor
    at::Tensor originalTensor;  // 原始 tensor
    int peerRank;               // 对端 rank
    int tag;                    // 消息标签
    int64_t seq;                // 序列号
    std::shared_ptr<std::atomic<bool>> completed;  // 完成标志
    std::shared_ptr<std::string> errorMsg;         // 错误信息
};
```

### 4.5 两种等待机制对比

| 特性 | 轮询 (Collective Worker) | 条件变量 (P2P Workers) |
|------|--------------------------|------------------------|
| CPU 占用 | 高（1个核心忙等） | 低（休眠等待） |
| 延迟 | **极低**（~微秒） | 较高（~几十微秒） |
| 适用场景 | 集合通信（性能敏感） | P2P 通信（频率较低） |
| 代码位置 | `mooncake_worker_thread.cpp` | `mooncake_backend.cpp` |

**为什么集合通信用轮询，P2P 用条件变量？**

1. **集合通信**：broadcast、allreduce 等操作是性能关键路径，延迟要求极高
2. **P2P 通信**：send/recv 通常用于控制消息，频率较低，可以牺牲一点延迟换取 CPU 效率

### 4.6 线程生命周期管理

#### 4.6.1 创建

```cpp
// 1. Collective Worker - 静态成员，程序启动时
MooncakeWorker MooncakeBackend::worker_;  // 全局唯一

// 2. Connection Poller - Backend 构造时
MooncakeBackend::MooncakeBackend(...) {
    std::thread([...] { connectionPoller(...); }).detach();
}

// 3. P2P Workers - Backend 构造末尾
startP2PWorker();
```

#### 4.6.2 销毁

```cpp
// mooncake_backend.cpp:229
MooncakeBackend::~MooncakeBackend() {
    stopP2PWorker();  // 停止 P2P 线程
    // ...
}

// mooncake_backend.cpp:771
void MooncakeBackend::stopP2PWorker() {
    if (p2pSendWorkerRunning_.load()) {
        p2pSendWorkerRunning_ = false;
        p2pSendQueueCv_.notify_all();  // 唤醒等待的线程
        p2pSendWorkerThread_.join();   // 等待线程结束
    }
    // recv worker 类似...
}
```

**注意：**
- `detach()` 的线程（Collective Worker, Connection Poller）无法主动停止
- `join()` 的线程（P2P Workers）可以优雅停止

### 4.7 线程安全机制

```cpp
// P2P 队列的同步
std::queue<P2POp> p2pSendQueue_;
std::mutex p2pSendQueueMutex_;           // 互斥锁
std::condition_variable p2pSendQueueCv_; // 条件变量
std::atomic<bool> p2pSendWorkerRunning_; // 原子标志

// Collective Worker 的任务状态
std::atomic<WorkerTaskStatus> task_status[kNumTasks_];  // 原子状态
volatile bool active;  // volatile 确保可见性
```

### 4.8 线程总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    每个进程的线程布局                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  主线程 ────────────────────────────────────────────────────    │
│    │                                                            │
│    ├─→ Collective Worker (静态，唯一，轮询)                      │
│    │     └─ 处理: broadcast, allreduce, allgather, etc.        │
│    │                                                            │
│    ├─→ Connection Poller (每 Backend 一个，sleep)               │
│    │     └─ 建立 RDMA 连接                                      │
│    │                                                            │
│    ├─→ P2P Send Worker (每 Backend 一个，条件变量)              │
│    │     └─ 处理: send 请求                                     │
│    │                                                            │
│    └─→ P2P Recv Worker (每 Backend 一个，条件变量)              │
│          └─ 处理: recv 请求                                     │
│                                                                 │
│  总计: 1 + 1 + 2*N 个后台线程 (N = Backend 数量)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.9 Collective Worker 执行流程详解

> 代码位置：`mooncake_worker_thread.cpp:14-222`

本节逐行解析 `startWorker()` 函数，这是 Collective Worker 线程的核心逻辑。

#### 4.9.1 线程初始化

```cpp
void MooncakeWorker::startWorker() {
    running_ = true;                          // 设置运行标志
    std::thread([this] {                      // 创建新线程，lambda 捕获 this
        std::atomic<WorkerTaskStatus> task_status[kNumTasks_];  // 4 个槽位的状态（线程局部）
        using clock = std::chrono::high_resolution_clock;
        clock::time_point activeTime[kNumTasks_];   // 4 个任务的激活时间戳（用于超时检测）
        TransferMetadata::NotifyDesc msg{"ping", "ping"};  // ping 消息，检测 peer 存活
```

**关键点：**
- `task_status[4]` 是**线程局部变量**，不是成员变量
- `activeTime[4]` 记录任务开始时间，用于超时判断
- `msg` 用于 ping 其他 rank，检测是否存活

#### 4.9.2 主循环与任务检测

```cpp
        while (running_) {
            PAUSE();                          // x86 pause 指令，降低功耗，仍是忙等待
            for (size_t i = 0; i < kNumTasks_; ++i) {  // 遍历 4 个任务槽位
                auto &task = tasks_[i];
                if (!task.active) {           // 任务未激活
                    task_status[i].store(IDLE, std::memory_order_release);
                    continue;
                }
```

**关键点：**
- `PAUSE()` 是 `_mm_pause()` 的包装，短暂暂停 CPU
- 1 个线程轮询 4 个槽位，不是 4 个线程

#### 4.9.3 判断是否跳过传输

```cpp
                auto group = (TransferGroupMeta *)task.transferGroupMeta;
                bool skipTransfer = (task.opType == c10d::OpType::BROADCAST &&
                                     group->rank != task.broadcastRoot) ||
                                    task.opType == c10d::OpType::BARRIER;
```

**跳过传输的情况：**
- **Broadcast**：只有 root 发送数据，其他 rank 跳过
- **Barrier**：不传数据，只同步信号

```
Broadcast 场景 (root=0):
  Rank 0: skipTransfer = false  → 执行 RDMA 传输
  Rank 1: skipTransfer = true   → 跳过传输，直接等信号
  Rank 2: skipTransfer = true   → 跳过传输，直接等信号
```

#### 4.9.4 状态 IDLE → TRANSFERRED_1：构建 RDMA 请求

##### 计算源地址 source

```cpp
uint64_t source = group->segmentDescs[group->rank]
                      ->buffers[task.bufferOffset]  // send_buffer
                      .addr;
switch (task.opType) {
    case c10d::OpType::BROADCAST:
    case c10d::OpType::ALLREDUCE:
    case c10d::OpType::ALLGATHER:
        break;  // source 不变，发送整个 buffer
    case c10d::OpType::ALLTOALL_BASE:
    case c10d::OpType::ALLTOALL:
    case c10d::OpType::_REDUCE_SCATTER_BASE:
        source += j * task.tensorSize;  // AllToAll: 发给 rank j 的是第 j 块
        break;
}
```

**AllToAll 源地址布局：**
```
send_buffer (AllToAll, 3 ranks):
┌─────────────┬─────────────┬─────────────┐
│ 给 rank 0   │ 给 rank 1   │ 给 rank 2   │
│ source+0    │ source+size │ source+2*size│
└─────────────┴─────────────┴─────────────┘
```

##### 计算目标地址 target_offset

```cpp
uint64_t target_offset = group->segmentDescs[j]
                             ->buffers[task.bufferOffset + 2]  // recv_buffer (+2 跳过 send 双缓冲)
                             .addr;
switch (task.opType) {
    case c10d::OpType::BROADCAST:
        break;  // 所有 rank 收到同样位置
    case c10d::OpType::ALLREDUCE:
    case c10d::OpType::ALLGATHER:
    case c10d::OpType::ALLTOALL:
        target_offset += group->rank * task.tensorSize;  // 写到目标 rank 的第 [my_rank] 块
        break;
}
```

**AllGather 目标地址布局：**
```
rank 0 的 recv_buffer:
┌─────────────┬─────────────┬─────────────┐
│ 来自 rank 0 │ 来自 rank 1 │ 来自 rank 2 │
│ offset+0    │ offset+size │ offset+2*size│
└─────────────┴─────────────┴─────────────┘
```

##### 提交 RDMA 请求

```cpp
entries.push_back(TransferRequest{
    .opcode = TransferRequest::WRITE,
    .source = (void *)source,
    .target_id = group->segmentIDs[j],
    .target_offset = target_offset,
    .length = task.tensorSize,
});
// ...
task.batchID = group->engine->allocateBatchID(entries.size());
group->engine->submitTransfer(task.batchID, entries);  // 提交 RDMA！
activeTime[i] = clock::now();
task_status[i].store(TRANSFERRED_1, std::memory_order_release);
```

#### 4.9.5 状态 TRANSFERRED_1 → SIGNALED_1：等待传输完成 + 发信号

##### 检查传输完成

```cpp
group->engine->getTransferStatus(task.batchID, task_id, status);
if (status.s != TransferStatusEnum::COMPLETED) {
    if (status.s == TransferStatusEnum::FAILED ||
        (diff.count() > kPingTimeoutMicroseconds_ &&
         group->engine->sendNotifyByName(...))) {
        // 失败或超时且 ping 不通 → 标记 peer 为 broken
        group->activeRanks[j] = false;
    } else {
        batch_done = false;  // 继续等待
    }
}
```

##### 发送完成信号

```cpp
auto source_ptr = (int32_t *)group->segmentDescs[group->rank]
                      ->buffers[task.bufferOffset + 4]  // cpu_sync_send_region
                      .addr;
*source_ptr = 1;  // 信号值
entries.push_back(TransferRequest{
    .opcode = TransferRequest::WRITE,
    .source = (void *)source_ptr,
    .target_id = group->segmentIDs[j],
    .target_offset = group->segmentDescs[j]
                         ->buffers[task.bufferOffset + 6]  // cpu_sync_recv_region
                         .addr + group->rank * sizeof(int32_t),  // 写到 [my_rank] 位置
    .length = sizeof(int32_t),
});
```

**信号机制示意（3 ranks）：**
```
初始状态:
  Rank 0 的 sync_recv: [0, 0, 0]
  Rank 1 的 sync_recv: [0, 0, 0]
  Rank 2 的 sync_recv: [0, 0, 0]

Rank 0 发送信号后:
  Rank 0 的 sync_recv: [1, 0, 0]  ← 自己写了 [0]
  Rank 1 的 sync_recv: [1, 0, 0]  ← Rank 0 写了 [0]
  Rank 2 的 sync_recv: [1, 0, 0]  ← Rank 0 写了 [0]

所有 rank 发完信号后:
  Rank 0 的 sync_recv: [1, 1, 1]  ← 全部收齐！
  Rank 1 的 sync_recv: [1, 1, 1]
  Rank 2 的 sync_recv: [1, 1, 1]
```

#### 4.9.6 状态 SIGNALED_1 → DONE：等待所有信号 + 调用 callback

```cpp
auto signal_ptr = (int32_t *)group->segmentDescs[group->rank]
                      ->buffers[task.bufferOffset + 6]  // cpu_sync_recv_region
                      .addr;
for (int j = 0; j < group->size; ++j) {
    if (group->activeRanks[j] && signal_ptr[j] != 1) {
        // 还没收到 rank j 的信号
        all_received = false;
    }
}
if (all_received) {
    for (int j = 0; j < group->size; ++j) {
        signal_ptr[j] = 0;  // 清零，为下次准备
    }
    task_status[i].store(DONE, std::memory_order_release);
    task.active = false;
    if (hasCallback_[i]) {
        callbacks_[i]();  // 调用回调 (bufferToTensor)
    }
}
```

#### 4.9.7 双缓冲机制与 Buffer 索引计算

每种 buffer 都有 `[0]` 和 `[1]` 两份，实现双缓冲流水线：

**Buffer 布局（每个 Backend 占 10 个槽位）：**

```
buffers[] 数组:
┌───────────────────────────────────────────────────────┐
│  [0]  send_buffer_[0]        ─┐                       │
│  [1]  send_buffer_[1]         ├─ 双缓冲              │
├───────────────────────────────────────────────────────┤
│  [2]  recv_buffer_[0]        ─┐                       │
│  [3]  recv_buffer_[1]         ├─ 双缓冲              │
├───────────────────────────────────────────────────────┤
│  [4]  cpu_sync_send_region_[0] ─┐                     │
│  [5]  cpu_sync_send_region_[1]   ├─ 双缓冲           │
├───────────────────────────────────────────────────────┤
│  [6]  cpu_sync_recv_region_[0] ─┐                     │
│  [7]  cpu_sync_recv_region_[1]   ├─ 双缓冲           │
├───────────────────────────────────────────────────────┤
│  [8]  warmup_send_region_                             │
│  [9]  warmup_recv_region_                             │
└───────────────────────────────────────────────────────┘
```

**bufferOffset 计算：**

```cpp
// mooncake_worker.cu:278
int bufferOffset = meta->bufferBaseIndex + meta->taskCount % 2;
//                 └── 基址 (0/10/20...)    └── 0 或 1（双缓冲选择）
```

**偏移量对照表：**

| 代码中的偏移 | 实际访问 | 说明 |
|-------------|----------|------|
| `bufferOffset + 0` | send_buffer_[N] | 发送数据 |
| `bufferOffset + 2` | recv_buffer_[N] | 接收数据（跳过 send 的双缓冲） |
| `bufferOffset + 4` | cpu_sync_send_[N] | 发送信号源 |
| `bufferOffset + 6` | cpu_sync_recv_[N] | 接收信号数组 |

**为什么 +2 而不是 +1？**

```
taskCount = 0 时，bufferOffset = 0:
  +0 → buffers[0] = send_buffer_[0]     ✓
  +2 → buffers[2] = recv_buffer_[0]     ✓  (跳过 [1])
  +4 → buffers[4] = sync_send_[0]       ✓
  +6 → buffers[6] = sync_recv_[0]       ✓

taskCount = 1 时，bufferOffset = 1:
  +0 → buffers[1] = send_buffer_[1]     ✓
  +2 → buffers[3] = recv_buffer_[1]     ✓
  +4 → buffers[5] = sync_send_[1]       ✓
  +6 → buffers[7] = sync_recv_[1]       ✓
```

**双缓冲的好处：**

```
时间线:
┌─────────────────────────────────────────────────────────────┐
│ 任务 0 (buffer A)      │ 任务 1 (buffer B)                  │
│ ├─ 传输中...           │ ├─ 同时准备数据到 buffer B         │
│ └─ 完成                │ └─ 开始传输                        │
└─────────────────────────────────────────────────────────────┘
               流水线重叠，提高吞吐量
```

#### 4.9.8 完整状态机总结

```
                    task.active = true (主线程设置)
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  IDLE                                                                     │
│  ├─ skipTransfer? ──YES──→ 直接进入 TRANSFERRED_1                         │
│  └─ NO: 构建 entries[]，submitTransfer()，进入 TRANSFERRED_1             │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  TRANSFERRED_1                                                            │
│  ├─ 轮询 getTransferStatus() 检查每个 RDMA 请求                           │
│  ├─ 超时 + ping 检测，标记 broken peer                                    │
│  └─ 全部完成: 发送信号 (写 1 到每个 rank 的 sync_recv[my_rank])           │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  SIGNALED_1                                                               │
│  ├─ 轮询检查 signal_ptr[j] == 1 对所有活跃 rank                           │
│  ├─ 超时 + ping 检测，标记 broken peer                                    │
│  └─ 收齐: 清零 signal_ptr[]，进入 DONE                                    │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  DONE                                                                     │
│  ├─ task.active = false                                                   │
│  └─ callbacks_[i]() → bufferToTensor() → 数据拷回用户 tensor              │
└──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    回到 IDLE，等待下一个任务
```

#### 4.9.9 各集合操作的 RDMA 模式对比

| 操作 | source 计算 | target_offset 计算 | RDMA 次数 |
|------|-------------|-------------------|-----------|
| **Broadcast** | 不变 | 不变 | root → 所有 rank |
| **AllReduce** | 不变 | `+rank*size` | 所有 rank → 所有 rank |
| **AllGather** | 不变 | `+rank*size` | 所有 rank → 所有 rank |
| **AllToAll** | `+j*size` | `+rank*size` | 每个 rank 发 N 块到 N 个 rank |
| **ReduceScatter** | `+j*size` | `+rank*size` | 所有 rank → 所有 rank |
| **Barrier** | 跳过传输 | 跳过传输 | 只发信号 |

---

## 5. 核心数据结构详解

> 本节详细介绍 mooncake-ep 中的核心数据结构及其字段含义。

### 5.1 TransferGroupMeta（通信组元数据）

> 定义位置：`mooncake_worker.cuh:20-39`

这个结构体存储了一个通信组（ProcessGroup）的所有元数据，是 Worker 线程执行任务时的核心上下文。

```cpp
struct TransferGroupMeta {
    // ============ 基本信息 ============
    int rank;           // 当前进程在组内的 rank (0 ~ size-1)
    int size;           // 组内总进程数
    int taskCount;      // 任务计数器，用于双缓冲切换 (taskCount % 2)

    // ============ 活跃 rank 管理 ============
    bool* activeRanks;           // CPU 端活跃 rank 数组 [size]
    bool* activeRanksDevice;     // GPU 端活跃 rank 数组（pinned memory）
    at::Tensor activeRanksTensor; // PyTorch Tensor 形式，方便 Python 访问
    bool peerConnected[kMaxNumRanks]{};  // 与每个 peer 的连接状态 [64]

    // ============ 传输引擎相关 ============
    TransferEngine* engine;      // RDMA 传输引擎指针
    c10::intrusive_ptr<::c10d::Store> store;  // PyTorch 分布式 Store

    // ============ 缓冲区索引 ============
    int bufferBaseIndex;         // 缓冲区基址索引 = backendIndex * 10
    int backendIndex;            // Backend 实例索引 (0, 1, 2...)

    // ============ RDMA 段信息 ============
    TransferMetadata::SegmentID segmentIDs[kMaxNumRanks];  // 每个 peer 的段 ID [64]
    std::shared_ptr<TransferMetadata::SegmentDesc> segmentDescs[kMaxNumRanks];  // 段描述符 [64]

    // ============ P2P 序列号管理（避免乱序） ============
    int64_t p2pSendSeq[kMaxNumRanks]{};           // 发送序列号 [64]
    int64_t p2pRecvSeq[kMaxNumRanks]{};           // 接收序列号 [64]
    int64_t p2pSendLowestInFlight[kMaxNumRanks]{}; // 最小在途发送序列号 [64]
    int64_t p2pRecvLowestInFlight[kMaxNumRanks]{}; // 最小在途接收序列号 [64]
    int64_t p2pRecvNextExpected[kMaxNumRanks]{};  // 下一个期望接收的序列号 [64]
};
```

**字段分组说明：**

| 分组 | 字段 | 用途 |
|------|------|------|
| **基本信息** | rank, size, taskCount | 标识进程和控制双缓冲 |
| **活跃管理** | activeRanks* | 支持弹性训练，动态标记存活 rank |
| **传输引擎** | engine, store | RDMA 操作和元数据交换 |
| **缓冲区** | bufferBaseIndex | 决定使用哪组 buffers[]（每个 Backend 占 10 个） |
| **RDMA 段** | segmentIDs, segmentDescs | 存储远端内存段信息 |
| **P2P 序列号** | p2p*Seq* | 保证 send/recv 操作的顺序性 |

**关键点：`bufferBaseIndex` 计算方式**

```cpp
// mooncake_backend.cpp:204
bufferBaseIndex = backendIndex_ * 10;
```

每个 Backend 占用 10 个 buffer 槽位：
- `[0-1]`: send_buffer_[0], send_buffer_[1]
- `[2-3]`: recv_buffer_[0], recv_buffer_[1]
- `[4-5]`: cpu_sync_send_region_[0], cpu_sync_send_region_[1]
- `[6-7]`: cpu_sync_recv_region_[0], cpu_sync_recv_region_[1]
- `[8]`: warmup_send_region_
- `[9]`: warmup_recv_region_

### 5.2 Task（任务结构体）

> 定义位置：`mooncake_worker.cuh:42-51`

这是 Worker 线程处理的基本单元，描述一个待执行的集合通信操作。

```cpp
__global__ struct Task {
    volatile bool active = false;     // 任务是否激活（主线程设置，Worker 读取）
    c10d::OpType opType = c10d::OpType::UNKNOWN;  // 操作类型
    size_t tensorSize;                // 数据大小（字节）
    int64_t broadcastRoot;            // Broadcast 的 root rank（仅 BROADCAST 有效）
    int bufferOffset;                 // 使用的缓冲区偏移（双缓冲选择）
    BatchID batchID;                  // RDMA 批次 ID（用于跟踪传输状态）
    void* transferGroupMeta;          // 指向 TransferGroupMeta 的指针
};
```

**字段详解：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `active` | `volatile bool` | **任务激活标志**。主线程置 true 表示有任务，Worker 处理完置 false。`volatile` 确保多线程可见性 |
| `opType` | `c10d::OpType` | 操作类型：BROADCAST, ALLREDUCE, ALLGATHER 等 |
| `tensorSize` | `size_t` | 要传输的数据大小（单位：字节） |
| `broadcastRoot` | `int64_t` | Broadcast 操作的源 rank。非 Broadcast 操作忽略此字段 |
| `bufferOffset` | `int` | 选择哪个缓冲区：`bufferBaseIndex + taskCount % 2` |
| `batchID` | `BatchID` | TransferEngine 分配的批次 ID，用于查询传输状态 |
| `transferGroupMeta` | `void*` | 指向 TransferGroupMeta，提供完整上下文 |

**`__global__` 关键字说明：**

```cpp
__global__ struct Task { ... };
```

这里 `__global__` 不是 CUDA kernel 标记，而是一个宏，用于标记这个结构体可以在 GPU 上访问（通过 pinned memory）：

```cpp
// 分配时使用 cudaHostAllocMapped
cudaHostAlloc(&tasks_, kNumTasks_ * sizeof(Task), cudaHostAllocMapped);
```

**任务生命周期：**

```
主线程                           Worker 线程
   │                                 │
   ├─ 填充 Task 字段                  │
   ├─ task.active = true ──────────→│ 检测到 active
   │                                 ├─ 读取 Task 信息
   ▼                                 ├─ 执行 RDMA 传输
  等待                               ├─ 等待完成信号
                                     ├─ 调用 callback
                                     └─ task.active = false
```

### 5.3 WorkerTaskStatus（任务状态枚举）

> 定义位置：`mooncake_worker_thread.cpp:10-13`

```cpp
enum WorkerTaskStatus {
    IDLE = 0,          // 空闲状态，无任务或任务未激活
    TRANSFERRED_1 = 1, // 数据传输完成，等待发送信号
    SIGNALED_1 = 2,    // 信号已发送，等待接收所有信号
    DONE = 3,          // 全部完成，准备调用 callback
};
```

**状态机流转：**

```
                ┌───────────────────────────────────────────────┐
                │                                               │
                ▼                                               │
            ┌──────┐     task.active      ┌──────────────┐     │
            │ IDLE │ ──────────────────→ │ TRANSFERRED_1 │     │
            └──────┘                      └──────────────┘     │
                ▲                                │              │
                │                                │ RDMA 完成    │
                │                                ▼              │
                │                         ┌──────────────┐     │
                │                         │  SIGNALED_1  │     │
                │                         └──────────────┘     │
                │                                │              │
                │                                │ 收到所有信号  │
                │                                ▼              │
                │                           ┌──────┐           │
                └───────────── callback ←── │ DONE │ ──────────┘
                                            └──────┘
```

---

## 6. MooncakeEpBuffer 与 IBGDA P2P 通信

> 本节详细介绍 MoE (Mixture of Experts) 场景下的高性能通信实现。

### 6.1 两套通信系统对比

Mooncake-EP 实际上有**两套独立的通信系统**：

| 特性 | MooncakeBackend | MooncakeEpBuffer |
|------|-----------------|------------------|
| **用途** | 标准 PyTorch 集合操作 | MoE dispatch/combine |
| **定义位置** | `mooncake_backend.cpp` | `mooncake_ep_buffer.cpp` |
| **RDMA 发起者** | **CPU** (TransferEngine) | **GPU 直接发起** (IBGDA) |
| **rank 限制** | kMaxNumRanks=64 | **无硬编码限制** |
| **通信模式** | 基于 Task 队列 + Worker 线程 | GPU kernel 直接发送 |
| **NVLink 支持** | 不支持 | 同节点 IPC memory |
| **延迟** | 较高（需 CPU 参与） | **极低**（GPU 直接发起） |

**为什么需要两套系统？**

MoE 模型的特点决定了标准集合操作不适用：
1. **细粒度通信**：每个 token 可能被路由到不同 expert
2. **All-to-All 模式**：不是简单的 broadcast/allreduce
3. **延迟敏感**：在 forward pass 中间进行，延迟直接影响吞吐

### 6.2 MooncakeEpBuffer 类结构

> 定义位置：`mooncake_ep_buffer.h:61-168`

```cpp
struct MooncakeEpBuffer {
private:
    // ============ 设备信息 ============
    int device_id;           // CUDA 设备 ID
    int rank, num_ranks;     // rank 信息（无 kMaxNumRanks 限制！）
    int clock_rate_khz;      // GPU 时钟频率（用于超时计算）

    // ============ GDR 缓冲区 ============
    int buffer_idx{};                 // 双缓冲索引
    int64_t num_ep_buffer_bytes;      // 缓冲区总大小
    void* gdr_buffer = nullptr;       // GPU Direct RDMA 缓冲区

    // ============ IBGDA 相关 ============
    static constexpr size_t CTRL_BUF_SIZE = 1024 * 1024 * 1024;  // 1GB 控制缓冲区
    void* ctrl_buf = nullptr;         // IBGDA 控制缓冲区（GPU 可访问）
    ibv_mr* mr;                       // IB 内存区域
    std::vector<mlx5gda_qp*> qps;     // QP 列表（MAX_QP_COUNT 个）
    ibv_gid gid;                      // IB GID
    void* raddrs = nullptr;           // 远端地址数组（GPU 端）
    void* rkeys = nullptr;            // 远端 key 数组（GPU 端）
    void* qp_devctxs = nullptr;       // QP device context（GPU 端）
    std::string device_name;          // IB 设备名
    bool is_roce_ = false;            // 是否使用 RoCE
    bool ibgda_disabled_ = false;     // IBGDA 是否禁用

    // ============ NVLink P2P ============
    int32_t* nvlink_available = nullptr;  // 哪些 rank 支持 NVLink [num_ranks]
    void** ipc_peer_ptrs_host = nullptr;  // IPC 指针（Host 端）
    void** ipc_peer_ptrs = nullptr;       // IPC 指针（Device 端）

    // ============ CUDA 流 ============
    at::cuda::CUDAStream comm_stream; // 通信专用流

    // ============ 工作空间 ============
    void* workspace = nullptr;        // 32MB 工作空间

public:
    // dispatch: 将 tokens 分发到各 expert
    std::tuple<...> dispatch(...);

    // combine: 将 expert 输出合并回原位置
    std::tuple<...> combine(...);
};
```

**关键设计：GPU 可直接访问的数据**

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU 可访问的数据                           │
├──────────────────────────────────────────────────────────────┤
│  raddrs[num_ranks]     → 每个 rank 的远端 GDR buffer 地址     │
│  rkeys[num_ranks]      → 每个 rank 的 RDMA rkey             │
│  qp_devctxs[MAX_QP_COUNT] → QP device context               │
│  nvlink_available[num_ranks] → NVLink 可用性标志            │
│  ipc_peer_ptrs[num_ranks]    → 同节点 IPC 指针              │
│  gdr_buffer            → 本地 GDR 缓冲区                     │
│  ctrl_buf              → IBGDA 控制缓冲区                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.3 IBGDA 初始化流程

> 代码位置：`mooncake_ep_buffer.cpp:312-399`

```cpp
int MooncakeEpBuffer::init_ibgda() {
    // 1. 查找指定的 IB 设备
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    for (int i = 0; i < num_devices; ++i) {
        if (device_name == ibv_get_device_name(dev_list[i])) {
            nic_id = i;
            break;
        }
    }

    // 2. 打开设备并获取 GID
    ibv_context* ctx = ibv_open_device(dev_list[nic_id]);
    ibv_query_gid(ctx, 1, 3, &gid);

    // 3. 分配保护域并注册 GPU 内存
    ibv_pd* pd = ibv_alloc_pd(ctx);
    mr = ibv_reg_mr(pd, gdr_buffer, num_ep_buffer_bytes,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);

    // 4. 分配 GPU 控制缓冲区并注册为 umem
    cudaMalloc(&ctrl_buf, CTRL_BUF_SIZE);
    mlx5dv_devx_umem* ctrl_buf_umem = mlx5dv_devx_umem_reg(
        ctx, ctrl_buf, CTRL_BUF_SIZE, IBV_ACCESS_LOCAL_WRITE);

    // 5. 创建 MAX_QP_COUNT 个 QP
    for (int i = 0; i < MAX_QP_COUNT; ++i) {
        mlx5gda_qp* qp = mlx5gda_create_rc_qp(...);
        mlx5gda_modify_rc_qp_rst2init(qp, 0);

        // 将 QP context 复制到 GPU 内存
        mlx5gda_qp_devctx qp_devctx = {
            .qpn = qp->qpn,
            .wqeid_mask = qp->num_wqebb - 1,
            .wq = (mlx5gda_wqebb*)(ctrl_buf + qp->wq_offset),
            .cq = (mlx5_cqe64*)(ctrl_buf + qp->send_cq->cq_offset),
            .dbr = (mlx5gda_wq_dbr*)(ctrl_buf + qp->dbr_offset),
            .bf = (char*)qp->uar->reg_addr,
        };
        cudaMemcpy(qp_devctxs + i * sizeof(mlx5gda_qp_devctx),
                   &qp_devctx, sizeof(mlx5gda_qp_devctx), cudaMemcpyHostToDevice);
        qps.push_back(qp);
    }
    return 0;
}
```

**IBGDA 架构图：**

```
┌────────────────────────────────────────────────────────────────────────┐
│                           GPU Memory                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                        gdr_buffer                                │  │
│  │  (注册为 IBV MR, 可被远端 RDMA 访问)                              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                        ctrl_buf (1GB)                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │  │
│  │  │ QP0 WQ/CQ   │  │ QP1 WQ/CQ   │  │ ...         │              │  │
│  │  │ DBR/BF      │  │ DBR/BF      │  │             │              │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐                       │
│  │ raddrs[] │  │ rkeys[]  │  │ qp_devctxs[]   │                       │
│  └──────────┘  └──────────┘  └────────────────┘                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                    │
                    │ GPU 线程直接访问
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          Mellanox NIC                                   │
│                                                                        │
│  GPU 通过写入 ctrl_buf 中的 WQ/DBR/BF 直接控制 NIC 发起 RDMA           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.4 三种数据传输路径

> 代码位置：`mooncake_ep_kernel.cu:276-300`

dispatch/combine kernel 根据目标 rank 选择最优路径：

```cpp
if (dst_rank != rank) {
    bool use_nvlink = nvlink_available[dst_rank] != 0;
    if (use_nvlink) {
        // ========== 路径 1: NVLink P2P ==========
        // 同节点 GPU，通过 IPC 内存直接拷贝
        size_t offset = (char *)dst_ptr - (char *)(mxa_buffer);
        void* peer_dst_ptr = (char *)ipc_peer_ptrs[dst_rank] + offset;
        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg,
                          dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
    } else {
        // ========== 路径 2: IBGDA RDMA ==========
        // 跨节点，GPU 直接发起 RDMA write
        if (lane_id == 0) {
            uint64_t req_rptr_actual = raddr_array[dst_rank] +
                                       ((char *)dst_ptr - (char *)(mxa_buffer));
            auto ctx = ctx_array + dst_rank * num_qp_per_rank +
                       dst_expert_local_idx % num_qp_per_rank;
            device_mutex_lock_system(&ctx->mutex);
            __mlx5gda_device_write_rdma_write_wqe(ctx, src_ptr,
                device_byteswap(rkey_array[rank]),
                req_rptr_actual,
                device_byteswap(rkey_array[dst_rank]),
                num_bytes_per_msg);
            __mlx5gda_device_post_send_db(ctx);
            device_mutex_unlock_system(&ctx->mutex);
        }
    }
} else {
    // ========== 路径 3: 本地拷贝 ==========
    // 同一 rank，直接内存拷贝
    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg,
                      dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
}
```

**三种路径对比：**

| 路径 | 条件 | 机制 | 延迟 |
|------|------|------|------|
| **NVLink P2P** | 同节点 + NVLink 可用 | CUDA IPC memory copy | **最低** |
| **IBGDA RDMA** | 跨节点 | GPU 直接发起 RDMA write | 低 |
| **本地拷贝** | dst_rank == rank | GPU 内存拷贝 | 极低 |

### 6.5 GPU 发起 RDMA 的核心函数

> 代码位置：`mooncake_ep_kernel.cu:92-114`

```cpp
// 构建 RDMA Write WQE (Work Queue Element)
static __device__ void __mlx5gda_device_write_rdma_write_wqe(
    struct mlx5gda_qp_devctx *ctx,
    uint64_t laddr, __be32 lkey,    // 本地地址和 key
    uint64_t raddr, __be32 rkey,    // 远端地址和 key
    uint32_t bytes) {               // 传输字节数

    // 获取下一个 WQE 槽位
    struct mlx5gda_rdma_write_wqe *wqe =
        (mlx5gda_rdma_write_wqe *)(ctx->wq + (ctx->wq_head & ctx->wqeid_mask));

    // 填充控制段
    ctrl_seg.qpn_ds = device_byteswap((ctx->qpn << 8) | 3);
    ctrl_seg.fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
    ctrl_seg.opmod_idx_opcode = device_byteswap(
        ((uint32_t)ctx->wq_head << 8) | MLX5_OPCODE_RDMA_WRITE);

    // 填充远端地址段
    raddr_seg.raddr = device_byteswap(raddr);
    raddr_seg.rkey = rkey;

    // 填充数据段
    data_seg.byte_count = device_byteswap(bytes);
    data_seg.lkey = lkey;
    data_seg.addr = device_byteswap(laddr);

    ++ctx->wq_head;
}

// 发送 Doorbell 通知 NIC
static __device__ void __mlx5gda_device_post_send_db(struct mlx5gda_qp_devctx *ctx) {
    // 1. 更新 DBR (Doorbell Record)
    ptx_st32_rel_sys_na(&ctx->dbr->send_counter, device_byteswap((uint32_t)ctx->wq_head));

    // 2. Ring Doorbell - 写入 BlueFlame 寄存器
    struct mlx5gda_wqebb *last_wqe = ctx->wq + ((ctx->wq_head - 1) & ctx->wqeid_mask);
    ptx_st64_rel_sys_na((uint64_t*)(ctx->bf + ctx->bf_offset), *(uint64_t*)last_wqe);

    // 3. 切换 BlueFlame offset
    ctx->bf_offset ^= MLX5GDA_BF_SIZE;
}
```

**RDMA 发送流程图：**

```
GPU 线程执行 dispatch kernel
        │
        ├─ 1. 获取 QP context: ctx_array[dst_rank * num_qp_per_rank + ...]
        │
        ├─ 2. 加锁: device_mutex_lock_system(&ctx->mutex)
        │
        ├─ 3. 构建 WQE:
        │      └─ __mlx5gda_device_write_rdma_write_wqe(ctx, laddr, lkey, raddr, rkey, size)
        │           │
        │           ├─ 填充 ctrl_seg (QPN, opcode)
        │           ├─ 填充 raddr_seg (远端地址)
        │           └─ 填充 data_seg (本地地址, 长度)
        │
        ├─ 4. 发送 Doorbell:
        │      └─ __mlx5gda_device_post_send_db(ctx)
        │           │
        │           ├─ 更新 DBR (send_counter)
        │           └─ 写入 BlueFlame 寄存器
        │
        ├─ 5. 解锁: device_mutex_unlock_system(&ctx->mutex)
        │
        └─ NIC 收到 Doorbell，执行 RDMA Write
```

### 6.6 NVLink IPC 初始化

> 代码位置：`mooncake_ep_buffer.cpp:482-563`

```cpp
void MooncakeEpBuffer::sync_nvlink_ipc_handles(
    const std::vector<std::vector<int32_t>>& remote_handles) {

    int device_count;
    cudaGetDeviceCount(&device_count);

    // 同节点 rank 范围
    int node_id = rank / device_count;
    int group_start = node_id * device_count;
    int group_end = std::min(group_start + device_count, num_ranks);

    for (int dst_rank = group_start; dst_rank < group_end; ++dst_rank) {
        if (dst_rank == rank) {
            // 本地 rank，直接使用本地指针
            ipc_peer_ptrs_host[dst_rank] = gdr_buffer;
            continue;
        }

        // 检查并启用 peer access
        int can_access_peer = 0;
        cudaDeviceCanAccessPeer(&can_access_peer, device_id, dst_rank % device_count);

        if (can_access_peer) {
            cudaDeviceEnablePeerAccess(dst_rank % device_count, 0);
            nvlink_array[dst_rank] = 1;

            // 打开远端 IPC handle
            cudaIpcMemHandle_t remote_handle;
            memcpy(&remote_handle, remote_handles[dst_rank].data(), sizeof(cudaIpcMemHandle_t));

            void* peer_ptr = nullptr;
            cudaIpcOpenMemHandle(&peer_ptr, remote_handle, cudaIpcMemLazyEnablePeerAccess);
            ipc_peer_ptrs_host[dst_rank] = peer_ptr;
        }
    }

    // 复制到 GPU 内存
    cudaMemcpy(nvlink_available, nvlink_array.data(), num_ranks * sizeof(int32_t), ...);
    cudaMemcpy(ipc_peer_ptrs, ipc_peer_ptrs_host, num_ranks * sizeof(void*), ...);
}
```

### 6.7 dispatch/combine 操作流程

**Dispatch（分发 tokens 到 experts）：**

```
输入: x[num_tokens, hidden], topk_idx[num_tokens, num_topk]
     └─ topk_idx 指定每个 token 路由到哪些 expert

┌────────────────────────────────────────────────────────────────────┐
│                        Dispatch Kernel                              │
├────────────────────────────────────────────────────────────────────┤
│  for each token:                                                    │
│      for each selected expert in topk_idx:                         │
│          dst_rank = expert_id / num_local_experts                  │
│          if dst_rank == self:                                       │
│              local copy                                             │
│          elif nvlink_available[dst_rank]:                          │
│              NVLink IPC copy                                        │
│          else:                                                      │
│              IBGDA RDMA write                                       │
│                                                                     │
│  发送信号通知接收方                                                   │
│  等待所有 rank 的信号                                                │
│  打包接收到的数据                                                    │
└────────────────────────────────────────────────────────────────────┘

输出: packed_recv_x[num_local_experts, num_ranks * max_tokens, hidden]
     └─ 每个 expert 收到的来自所有 rank 的 tokens
```

**Combine（合并 expert 输出）：**

```
输入: x[num_local_experts, num_ranks * max_tokens, hidden]
     topk_weights[num_tokens, num_topk]

┌────────────────────────────────────────────────────────────────────┐
│                        Combine Kernel                               │
├────────────────────────────────────────────────────────────────────┤
│  for each expert output:                                            │
│      根据 src_info 确定原始 token 位置                               │
│      发送回原始 rank (NVLink/IBGDA/local)                           │
│                                                                     │
│  发送信号通知接收方                                                   │
│  等待所有数据到达                                                    │
│  加权合并: combined[i] = Σ (weight[k] * expert_output[k])           │
└────────────────────────────────────────────────────────────────────┘

输出: combined_x[num_tokens, hidden]
     └─ 加权合并后的最终输出
```

### 6.8 性能优势分析

**传统 CPU 发起 RDMA vs IBGDA：**

```
传统方式 (MooncakeBackend):
GPU ──sync──→ CPU ──RDMA──→ NIC ──network──→ Remote
            │      │
            │      └── CPU 构建 WQE, 发 doorbell
            └────────── GPU-CPU 同步开销

IBGDA 方式 (MooncakeEpBuffer):
GPU ────────────────────RDMA────────────→ NIC ──network──→ Remote
     │
     └── GPU 线程直接构建 WQE, 发 doorbell
         无 GPU-CPU 同步！
```

**延迟对比：**

| 操作 | 传统方式 | IBGDA |
|------|----------|-------|
| GPU-CPU 同步 | ~10-100 μs | **0** |
| WQE 构建 | CPU 执行 | GPU 执行 |
| Doorbell | CPU 发 | GPU 发 |
| 适合场景 | 大批量传输 | **细粒度、低延迟** |

### 6.9 Python 接口

> 代码位置：`mooncake-integration/ep/ep_py.cpp:114-130`

```cpp
py::class_<MooncakeEpBuffer>(m, "Buffer")
    .def(py::init<int, int, int64_t, std::string>())  // rank, num_ranks, buffer_size, device_name
    .def("ibgda_disabled", &MooncakeEpBuffer::ibgda_disabled)
    .def("is_roce", &MooncakeEpBuffer::is_roce)
    .def("sync_ib", &MooncakeEpBuffer::sync_ib)
    .def("sync_roce", &MooncakeEpBuffer::sync_roce)
    .def("get_mr_info", &MooncakeEpBuffer::get_mr_info)
    .def("get_gid", &MooncakeEpBuffer::get_gid)
    .def("get_local_qpns", &MooncakeEpBuffer::get_local_qpns)
    .def("get_local_lids", &MooncakeEpBuffer::get_local_lids)
    .def("get_ipc_handle", &MooncakeEpBuffer::get_ipc_handle)
    .def("sync_nvlink_ipc_handles", &MooncakeEpBuffer::sync_nvlink_ipc_handles)
    .def("dispatch", &MooncakeEpBuffer::dispatch)
    .def("combine", &MooncakeEpBuffer::combine)
    .def("get_next_combine_buffer", &MooncakeEpBuffer::get_next_combine_buffer);
```

**Python 使用示例：**

```python
import mooncake.ep as ep

# 创建 Buffer
buffer = ep.Buffer(rank=0, num_ranks=8, buffer_size=1024*1024*1024, device_name="mlx5_0")

# 同步连接信息
if buffer.is_roce():
    buffer.sync_roce(remote_addrs, remote_keys, remote_qpns, subnet_prefixes, interface_ids)
else:
    buffer.sync_ib(remote_addrs, remote_keys, remote_qpns, remote_lids)

# 同步 NVLink handles
buffer.sync_nvlink_ipc_handles(all_ipc_handles)

# 执行 dispatch
packed_x, scales, count, src_info, layout_range, event, hook = buffer.dispatch(
    x, topk_idx, active_ranks, max_tokens, num_experts, timeout_us, use_fp8, async_, return_hook
)

# 执行 combine
combined_x, event, hook = buffer.combine(
    expert_out, topk_idx, topk_weights, src_info, layout_range,
    active_ranks, max_tokens, num_experts, timeout_us, zero_copy, async_, return_hook
)
```

---

*待续：更多文件的详细解读将陆续添加...*
