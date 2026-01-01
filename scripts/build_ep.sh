#!/bin/bash
# =============================================================================
# Mooncake EP 编译脚本
# 用于从源码编译 mooncake-ep 并运行测试
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EP_DIR="$PROJECT_ROOT/mooncake-ep"
WHEEL_DIR="$PROJECT_ROOT/mooncake-wheel"
TEST_DIR="$WHEEL_DIR/tests"

# =============================================================================
# 步骤 0: 环境检查
# =============================================================================
check_environment() {
    log_info "检查编译环境..."

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3"
        exit 1
    fi
    log_ok "Python: $(python3 --version)"

    # 检查 PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        log_error "未找到 PyTorch"
        exit 1
    fi
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    log_ok "PyTorch: $TORCH_VERSION (CUDA $CUDA_VERSION)"

    # 检查 CUDA
    if ! command -v nvcc &> /dev/null; then
        log_warn "未找到 nvcc，将使用 PyTorch 自带的 CUDA"
    else
        log_ok "NVCC: $(nvcc --version | grep release | awk '{print $6}')"
    fi

    # 检查 RDMA 库
    if ! ldconfig -p | grep -q libibverbs; then
        log_warn "未找到 libibverbs，RDMA 功能可能不可用"
    else
        log_ok "libibverbs: 已安装"
    fi

    if ! ldconfig -p | grep -q libmlx5; then
        log_warn "未找到 libmlx5，IBGDA 功能可能不可用"
    else
        log_ok "libmlx5: 已安装"
    fi

    # 检查 Transfer Engine 依赖
    if [ ! -f "$WHEEL_DIR/mooncake/engine.so" ]; then
        log_error "未找到 $WHEEL_DIR/mooncake/engine.so"
        log_error "请先编译 mooncake-transfer-engine:"
        echo "  cd $PROJECT_ROOT && mkdir -p build && cd build"
        echo "  cmake -DWITH_TE=ON -DCMAKE_BUILD_TYPE=Release .."
        echo "  make -j\$(nproc) install"
        exit 1
    fi
    log_ok "engine.so: 存在"

    if [ ! -f "$WHEEL_DIR/mooncake/libtransfer_engine.so" ]; then
        log_error "未找到 $WHEEL_DIR/mooncake/libtransfer_engine.so"
        exit 1
    fi
    log_ok "libtransfer_engine.so: 存在"
}

# =============================================================================
# 步骤 1: 清理旧的编译产物
# =============================================================================
clean_build() {
    log_info "清理旧的编译产物..."

    cd "$EP_DIR"
    rm -rf build/ dist/ *.egg-info/
    rm -f mooncake/*.so

    # 清理 wheel 目录中的旧 ep 模块
    rm -f "$WHEEL_DIR/mooncake/ep_*.so"

    log_ok "清理完成"
}

# =============================================================================
# 步骤 2: 编译 mooncake-ep
# =============================================================================
build_ep() {
    log_info "开始编译 mooncake-ep..."

    cd "$EP_DIR"

    # 获取 PyTorch 版本后缀
    TORCH_VERSION_SUFFIX=$(python3 -c "
import re, torch
v = re.match(r'\d+(?:\.\d+)*', torch.__version__).group()
print('_' + v.replace('.', '_'))
")
    log_info "PyTorch 版本后缀: $TORCH_VERSION_SUFFIX"

    # 编译
    log_info "运行 setup.py build_ext --inplace ..."
    python3 setup.py build_ext --inplace 2>&1 | tee /tmp/mooncake_ep_build.log

    # 检查编译产物
    EP_SO="$EP_DIR/mooncake/ep${TORCH_VERSION_SUFFIX}.cpython-*-linux-gnu.so"
    if ls $EP_SO 1> /dev/null 2>&1; then
        log_ok "编译成功: $(ls $EP_SO)"
    else
        log_error "编译失败，请检查 /tmp/mooncake_ep_build.log"
        exit 1
    fi

    # 复制到 mooncake-wheel 目录
    log_info "复制 .so 文件到 mooncake-wheel/mooncake/ ..."
    cp -v $EP_SO "$WHEEL_DIR/mooncake/"

    log_ok "mooncake-ep 编译完成"
}

# =============================================================================
# 步骤 3: 安装 mooncake-wheel (editable mode)
# =============================================================================
install_wheel() {
    log_info "安装 mooncake-wheel (editable mode)..."

    cd "$WHEEL_DIR"
    pip install -e . --quiet

    log_ok "mooncake-wheel 安装完成"
}

# =============================================================================
# 步骤 4: 验证安装
# =============================================================================
verify_install() {
    log_info "验证安装..."

    python3 -c "
import mooncake.ep
print('mooncake.ep 模块加载成功')

# 检查关键函数
from mooncake.mooncake_ep_buffer import Buffer
print('Buffer 类导入成功')
"

    if [ $? -eq 0 ]; then
        log_ok "验证通过"
    else
        log_error "验证失败"
        exit 1
    fi
}

# =============================================================================
# 步骤 5: 运行测试
# =============================================================================
run_test() {
    log_info "运行测试: test_mooncake_ep.py"

    cd "$TEST_DIR"

    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

    log_info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    python3 test_mooncake_ep.py

    if [ $? -eq 0 ]; then
        log_ok "测试通过"
    else
        log_error "测试失败"
        exit 1
    fi
}

# =============================================================================
# 主流程
# =============================================================================
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --check      仅检查环境"
    echo "  --clean      仅清理编译产物"
    echo "  --build      仅编译 (不安装/测试)"
    echo "  --install    编译并安装 (不测试)"
    echo "  --test       仅运行测试"
    echo "  --all        完整流程: 检查 -> 清理 -> 编译 -> 安装 -> 测试"
    echo "  --help       显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 --all              # 完整编译和测试"
    echo "  $0 --build --test     # 编译后直接测试"
    echo "  $0 --clean --build    # 清理后重新编译"
}

# 默认行为: 完整流程
if [ $# -eq 0 ]; then
    check_environment
    clean_build
    build_ep
    install_wheel
    verify_install
    log_ok "编译完成！运行测试请使用: $0 --test"
    exit 0
fi

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            check_environment
            shift
            ;;
        --clean)
            clean_build
            shift
            ;;
        --build)
            build_ep
            shift
            ;;
        --install)
            check_environment
            clean_build
            build_ep
            install_wheel
            verify_install
            shift
            ;;
        --test)
            run_test
            shift
            ;;
        --all)
            check_environment
            clean_build
            build_ep
            install_wheel
            verify_install
            run_test
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

log_ok "脚本执行完成"
