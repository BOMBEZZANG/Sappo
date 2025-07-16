# Sappo Trading Bot - Complete RL Integration

A comprehensive GUI-driven desktop application that manages the end-to-end workflow from data preprocessing to reinforcement learning model training and evaluation.

## 🚀 Features

### **Data Preprocessing Tab**
- Multi-asset, multi-timeframe data loading
- Unified dataset creation with 527+ features
- Real-time progress tracking
- Automatic results folder creation

### **RL Training & Evaluation Tab**
- **Data Selection**: Load preprocessed `.npy` files
- **Hyperparameter Configuration**: Learning rate, gamma, reward weights
- **Training Pipeline**: GRU-based PPO agent with validation
- **Real-time Progress Monitoring**: Live evaluation table updated every 10k steps
- **Model Evaluation**: Comprehensive performance analysis
- **Visualization**: Integrated performance charts
- **TensorBoard Integration**: Real-time training monitoring

## 📦 Installation

### Basic Dependencies (Required)
```bash
pip install pandas numpy scikit-learn matplotlib
```

### RL Dependencies (For Training)
```bash
pip install gymnasium stable-baselines3 torch tensorboard
```

### Complete Installation
```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Method 1: Integrated Launcher (Recommended)
```bash
python Scripts/PreProcessing/run_sappo_integrated.py
```

### Method 2: Direct Launch
```bash
cd Scripts/PreProcessing
python main_integrated.py
```

## 📊 Complete Workflow

### 1. Data Preprocessing
1. **Select CSV Files**: Choose your multi-timeframe crypto data
2. **Start Preprocessing**: Unify all coins into single dataset
3. **Output**: Unified `.npy` file with 527+ features

### 2. RL Training
1. **Select Data**: Load preprocessed `.npy` file
2. **Configure Parameters**: Set hyperparameters and reward weights
3. **Start Training**: Train GRU-based PPO agent
4. **Monitor Progress**: 
   - **Training Log**: Real-time training messages
   - **Training Progress**: Live evaluation table every 10k steps
   - **TensorBoard**: Advanced metrics visualization

### 3. Model Evaluation
1. **Evaluate Model**: Test on unseen data
2. **View Results**: Comprehensive performance metrics
3. **Performance Chart**: Portfolio progression visualization

## 🧠 RL Architecture

### **Environment (TradingEnv)**
- **State Space**: (24, 529) - 24-hour lookback with 529 features
- **Action Space**: Discrete(3) - Hold, Buy, Sell
- **Reward Function**: Multi-component reward with configurable weights

### **Agent (GRU-based PPO)**
- **Feature Extractor**: Dual GRU layers (128→64 units)
- **Policy Network**: Actor-Critic architecture
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable-Baselines3 + PyTorch

### **Training Pipeline**
- **Data Split**: 70% Train, 15% Validation, 15% Test
- **Validation**: Automatic best model saving based on Sharpe ratio
- **Real-time Monitoring**: 
  - **Every 10k Steps**: Validation evaluation with structured results
  - **Live Dashboard**: Evaluation table with metrics progression
  - **Best Tracking**: Automatic highlighting of best Sharpe ratio
  - **TensorBoard**: Advanced metrics and loss curves

## 🎯 Key Performance Metrics

### **Return Metrics**
- Total Return %
- Annualized Return
- Sharpe Ratio
- Sortino Ratio

### **Risk Metrics**
- Maximum Drawdown
- Volatility
- Calmar Ratio

### **Trading Metrics**
- Win Rate
- Profit Factor
- Trade Count
- Transaction Costs

## 📁 Project Structure

```
Scripts/PreProcessing/
├── main_integrated.py       # Integrated GUI application
├── run_sappo_integrated.py  # Launcher with dependency checking
├── preprocessing.py         # Data preprocessing pipeline
├── TradingEnv.py           # Custom trading environment
├── agent.py                # GRU-based trading agent
├── train.py                # Training pipeline
├── evaluate.py             # Evaluation and reporting
├── models/                 # Saved models directory
├── results/                # Preprocessed data directory
├── evaluation_results/     # Evaluation outputs
└── tensorboard_logs/       # TensorBoard logs
```

## ⚙️ Configuration

### **Hyperparameters**
- **Learning Rate**: 0.0001 (default)
- **Gamma**: 0.99 (discount factor)
- **Training Steps**: 100,000 (default)

### **Reward Weights**
- **Profit**: 1.0 (portfolio return weight)
- **Sharpe**: 0.5 (risk-adjusted return weight)
- **Cost**: 1.0 (transaction cost penalty)
- **MDD**: 0.5 (maximum drawdown penalty)

## 📊 Real-time Training Progress

### **Evaluation Progress Table**
During training, every 10,000 timesteps, the system performs validation and displays:

| Timestep | Mean Reward | Std Reward | Sharpe Ratio | Total Return | Max Drawdown | Final Value | Trade Count |
|----------|-------------|------------|--------------|--------------|--------------|-------------|-------------|
| 10,000   | -0.0020     | 0.0150     | -0.1340      | -5.00%       | 8.00%        | $9,500      | 15          |
| 20,000   | 0.0010      | 0.0120     | 0.0830       | 2.00%        | 6.00%        | $10,200     | 22          |
| 30,000   | 0.0050      | 0.0180     | 0.2780 ⭐    | 8.00%        | 4.00%        | $10,800     | 31          |

**Features:**
- **Live Updates**: Real-time progress during training
- **Best Highlighting**: Star (⭐) marks new best Sharpe ratios
- **Auto-scrolling**: Always shows latest evaluations
- **Thread-safe**: Updates from training thread to GUI safely

### **Progress Monitoring Benefits**
- **Early Stopping**: Identify optimal training duration
- **Hyperparameter Tuning**: See immediate impact of changes
- **Performance Trends**: Track improvement over time
- **Best Model Tracking**: Always know when best model was saved

## 🔧 Advanced Usage

### Command Line Training
```bash
python train.py preprocessed_data.npy 200000
```

### Command Line Evaluation
```bash
python evaluate.py models/best_model.zip test_data.npy
```

### TensorBoard Monitoring
```bash
tensorboard --logdir=tensorboard_logs
```

## 📈 Example Results

```
=== SAPPO Trading Bot Evaluation Report ===
📊 PERFORMANCE SUMMARY
Total Return: 15.23%
Annualized Return: 45.67%
Sharpe Ratio: 1.234
Max Drawdown: 8.45%

🎯 TRADING METRICS
Win Rate: 64.2%
Profit Factor: 1.67
Total Trades: 156

🏆 BENCHMARK COMPARISON
Agent Return: 15.23%
Buy & Hold Return: 8.91%
Outperformance: 6.32%
```

## 🐛 Troubleshooting

### **RL Dependencies Missing**
- App runs in preprocessing-only mode
- Install: `pip install gymnasium stable-baselines3 torch tensorboard`

### **GPU Support**
- PyTorch will automatically use GPU if available
- Check: `torch.cuda.is_available()`

### **Memory Issues**
- Reduce training steps or batch size
- Use smaller window size in preprocessing

### **Performance Issues**
- Enable GPU acceleration
- Reduce number of parallel environments
- Optimize hyperparameters

## 🔄 Model Lifecycle

1. **Data Collection** → CSV files from exchanges
2. **Preprocessing** → Unified feature engineering
3. **Training** → RL agent optimization
4. **Validation** → Performance monitoring
5. **Evaluation** → Backtesting on unseen data
6. **Deployment** → Model ready for live trading

## 🎓 Next Steps

- **Live Trading Integration**: Connect to exchange APIs
- **Portfolio Management**: Multi-asset position sizing
- **Advanced Strategies**: Option strategies, arbitrage
- **Risk Management**: Dynamic position sizing, stop-losses
- **Ensemble Methods**: Multiple agent strategies

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Environment](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)

---

**Built with ❤️ for algorithmic trading**