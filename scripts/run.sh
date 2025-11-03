#!/bin/bash

# ================================
# Transformer机器翻译项目一键运行脚本
# 自动设置环境、安装依赖、训练模型
# ================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 $1 未找到，请先安装"
        exit 1
    fi
}

# 检查GPU可用性
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        if [ "$GPU_COUNT" -gt 0 ]; then
            log_info "检测到 $GPU_COUNT 个GPU"
            return 0
        else
            log_warn "未检测到可用GPU，将使用CPU训练（速度较慢）"
            return 1
        fi
    else
        log_warn "无法检测GPU，将使用CPU训练"
        return 1
    fi
}

# 创建虚拟环境
create_venv() {
    if [ ! -d "venv" ]; then
        log_step "创建Python虚拟环境..."
        python -m venv venv
    fi
    
    log_step "激活虚拟环境..."
    source venv/bin/activate
}

# 安装依赖
install_dependencies() {
    log_step "安装Python依赖..."
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装PyTorch（根据是否有GPU选择版本）
    if check_gpu; then
        log_info "安装GPU版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "安装CPU版本的PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whnpu/cpu
    fi
    
    # 安装其他依赖
    pip install -r requirements.txt
    
    # 安装项目特定依赖
    pip install datasets tokenizers torchmetrics tensorboard
}

# 准备数据
prepare_data() {
    log_step "准备训练数据..."
    
    # 创建必要的目录
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p results
    
    # 检查数据目录是否存在
    if [ ! -d "data" ]; then
        log_info "创建数据目录..."
        mkdir -p data
    fi
}

# 训练模型
train_model() {
    local config_file=$1
    local seed=$2
    local num_epochs=$3
    
    log_step "开始训练Transformer模型..."
    log_info "配置文件: $config_file"
    log_info "随机种子: $seed"
    log_info "训练轮数: $num_epochs"
    
    # 设置随机种子环境变量
    export PYTHONHASHSEED=$seed
    
    python train.py \
        --config $config_file \
        --seed $seed \
        --num_epochs $num_epochs
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        log_info "模型训练完成！"
    else
        log_error "模型训练失败"
        exit 1
    fi
}

# 启动TensorBoard
start_tensorboard() {
    local port=$1
    
    log_step "启动TensorBoard（端口: $port）..."
    log_info "在浏览器中打开: http://localhost:$port"
    
    # 在后台启动TensorBoard
    tensorboard --logdir logs --port $port --bind_all &
    TENSORBOARD_PID=$!
    
    echo $TENSORBOARD_PID > tensorboard.pid
    log_info "TensorBoard进程ID: $TENSORBOARD_PID"
}

# 停止TensorBoard
stop_tensorboard() {
    if [ -f "tensorboard.pid" ]; then
        local pid=$(cat tensorboard.pid)
        if kill -0 $pid 2>/dev/null; then
            log_step "停止TensorBoard进程..."
            kill $pid
            rm tensorboard.pid
        fi
    fi
}

# 显示帮助信息
show_help() {
    echo "Transformer机器翻译项目一键运行脚本"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -c, --config FILE   指定配置文件（默认: config.yaml）"
    echo "  -s, --seed SEED     设置随机种子（默认: 42）"
    echo "  -e, --epochs EPOCHS 训练轮数（默认: 20）"
    echo "  -p, --port PORT     TensorBoard端口（默认: 6006）"
    echo "  --no-tensorboard    不启动TensorBoard"
    echo "  --no-venv           不使用虚拟环境"
    echo ""
    echo "示例:"
    echo "  $0                              # 使用默认参数运行"
    echo "  $0 -c config.yaml -s 123 -e 50  # 自定义配置、种子和轮数"
    echo "  $0 --no-tensorboard             # 不启动TensorBoard"
}

# 主函数
main() {
    # 默认参数
    local config_file="config.yaml"
    local seed=42
    local num_epochs=20
    local tensorboard_port=6006
    local use_tensorboard=true
    local use_venv=true
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                config_file="$2"
                shift 2
                ;;
            -s|--seed)
                seed="$2"
                shift 2
                ;;
            -e|--epochs)
                num_epochs="$2"
                shift 2
                ;;
            -p|--port)
                tensorboard_port="$2"
                shift 2
                ;;
            --no-tensorboard)
                use_tensorboard=false
                shift
                ;;
            --no-venv)
                use_venv=false
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    log_info "项目根目录: $PROJECT_ROOT"
    cd "$PROJECT_ROOT"
    
    # 检查必要命令
    log_step "检查系统依赖..."
    check_command python
    check_command pip
    
    # 检查Python版本
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
    
    # 创建和使用虚拟环境
    if [ "$use_venv" = true ]; then
        create_venv
    else
        log_warn "跳过虚拟环境创建"
    fi
    
    # 安装依赖
    install_dependencies
    
    # 准备数据
    prepare_data
    
    # 检查配置文件是否存在
    if [ ! -f "$config_file" ]; then
        log_error "配置文件 $config_file 不存在"
        log_info "请创建配置文件或使用 -c 参数指定正确的配置文件"
        exit 1
    fi
    
    # 启动TensorBoard
    if [ "$use_tensorboard" = true ]; then
        start_tensorboard $tensorboard_port
    fi
    
    # 训练模型
    train_model "$config_file" "$seed" "$num_epochs"
    
    # 停止TensorBoard
    if [ "$use_tensorboard" = true ]; then
        stop_tensorboard
    fi
    
    log_info "🎉 项目运行完成！"
    log_info "📊 模型权重保存在: checkpoints/"
    log_info "📈 训练日志在: logs/"
}

# 清理函数（在脚本退出时调用）
cleanup() {
    log_info "执行清理操作..."
    stop_tensorboard
}

# 设置陷阱，确保脚本退出时执行清理
trap cleanup EXIT

# 运行主函数
main "$@"
