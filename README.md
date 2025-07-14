# Dopamine Trading System

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

**A sophisticated AI-powered trading system with reinforcement learning, genetic algorithms, and adaptive neural networks**

</div>

## 🧠 Overview

The Dopamine Trading System is a cutting-edge AI-powered trading platform that combines reinforcement learning, genetic algorithms, and adaptive neural networks to make intelligent trading decisions. Built with clean architecture principles, the system features 5 specialized AI subsystems working in harmony to analyze market patterns and execute trades through NinjaTrader integration.

### ✨ Key Features

- **🤖 Reinforcement Learning Agent**: Deep Q-Network (DQN) with experience replay and dopamine-inspired learning
- **🧬 Genetic Algorithm Evolution**: DNA subsystem that evolves trading patterns over time
- **📊 Market Regime Detection**: Microstructure analysis without traditional indicators
- **🎯 Adaptive Neural Networks**: Self-modifying architectures that grow/shrink based on performance
- **🛡️ Immune System Risk Management**: Anomaly detection and risk assessment
- **⏰ Temporal Pattern Recognition**: Cycle detection and time-based market analysis
- **🔄 Multi-Objective Reward System**: Comprehensive reward calculation for optimal learning
- **📈 NinjaTrader Integration**: Seamless TCP bridge for live trading

## 🏗️ Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Dopamine Trading System                                │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   NinjaTrader   │◄──►│   TCP Bridge    │◄──►│  Market Data    │            │
│  │   C# Strategy   │    │   (Port 5556)   │    │   Processor     │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                           │                     │
│                                                           ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                        AI Subsystem Manager                                │ │
│  │                                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│  │  │    DNA      │ │  Temporal   │ │   Immune    │ │Microstructure│         │ │
│  │  │ Subsystem   │ │ Subsystem   │ │ Subsystem   │ │ Subsystem   │         │ │
│  │  │             │ │             │ │             │ │             │         │ │
│  │  │ • Genetic   │ │ • Cycle     │ │ • Anomaly   │ │ • Regime    │         │ │
│  │  │   Algorithm │ │   Detection │ │   Detection │ │   Detection │         │ │
│  │  │ • Pattern   │ │ • Time      │ │ • Risk      │ │ • Order     │         │ │
│  │  │   Evolution │ │   Patterns  │ │   Assessment│ │   Flow      │         │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │ │
│  │                                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                     Dopamine Subsystem                                 │ │ │
│  │  │                                                                         │ │ │
│  │  │  • Reward Prediction & Optimization                                    │ │ │
│  │  │  • Surprise Detection & Learning Enhancement                           │ │ │
│  │  │  • Prediction Error-Based Dopamine Signals                            │ │ │
│  │  └─────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                    Reinforcement Learning Agent                            │ │
│  │                                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────────────┐   │ │
│  │  │   Neural    │ │  Adaptive   │ │         Specialized Networks         │   │ │
│  │  │  Network    │ │  Network    │ │                                     │   │ │
│  │  │  Manager    │ │             │ │  • DQN Network                      │   │ │
│  │  │             │ │ • Dynamic   │ │  • Actor-Critic Network             │   │ │
│  │  │ • Model     │ │   Growth    │ │  • Prediction Networks              │   │ │
│  │  │   Lifecycle │ │ • Pruning   │ │  • Classification Networks          │   │ │
│  │  │ • Training  │ │ • Layer     │ │  • Ensemble Networks                │   │ │
│  │  │   Mgmt      │ │   Adaptation│ │                                     │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                          Trading Decisions                                 │ │
│  │                                                                             │ │
│  │  • Action Selection (Buy/Sell/Hold)                                        │ │
│  │  • Confidence Scoring                                                      │ │
│  │  • Position Sizing                                                         │ │
│  │  • Risk Management                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                   Data Flow Pipeline                        │
                    └─────────────────────────────────────────────────────────────┘

 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │ NinjaTrader │───►│ TCP Bridge  │───►│Market Data  │───►│ Feature     │
 │   C# Code   │    │ (Port 5556) │    │ Processor   │    │ Builder     │
 └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                                       │                 │
       │ Position/Account Data                 │ Real-time       │ AI Features
       │                                       │ Market Data     │
       ▼                                       ▼                 ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           State Vector                                     │
 │                                                                             │
 │  Prices │ Volumes │ Account │ Market │ Technical │ Subsystem │ Timestamp  │
 │   [50]  │   [50]  │   [10]  │   [5]  │    [20]   │    [25]   │     [1]    │
 └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     AI Subsystem Processing                                 │
 │                                                                             │
 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
 │  │    DNA      │  │  Temporal   │  │   Immune    │  │Microstructure│       │
 │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │       │
 │  │ │Pattern  │ │  │ │ Cycle   │ │  │ │Anomaly  │ │  │ │ Regime  │ │       │
 │  │ │Analysis │ │  │ │Detection│ │  │ │Detector │ │  │ │Detector │ │       │
 │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │       │
 │  │      │      │  │      │      │  │      │      │  │      │      │       │
 │  │      ▼      │  │      ▼      │  │      ▼      │  │      ▼      │       │
 │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │       │
 │  │ │ Signal  │ │  │ │ Signal  │ │  │ │ Signal  │ │  │ │ Signal  │ │       │
 │  │ │   Gen   │ │  │ │   Gen   │ │  │ │   Gen   │ │  │ │   Gen   │ │       │
 │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │       │
 │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
 │         │                 │                 │                 │           │
 │         └─────────────────┼─────────────────┼─────────────────┘           │
 │                           │                 │                             │
 │  ┌─────────────────────────────────────────────────────────────────────┐   │
 │  │                    Dopamine Subsystem                              │   │
 │  │                                                                     │   │
 │  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │   │
 │  │  │   Reward    │    │  Surprise   │    │  Learning   │             │   │
 │  │  │ Prediction  │    │  Detection  │    │Enhancement  │             │   │
 │  │  └─────────────┘    └─────────────┘    └─────────────┘             │   │
 │  └─────────────────────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    Signal Aggregation                                      │
 │                                                                             │
 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
 │  │    DNA      │  │  Temporal   │  │   Immune    │  │Microstructure│       │
 │  │   Signal    │  │   Signal    │  │   Signal    │  │   Signal    │       │
 │  │             │  │             │  │             │  │             │       │
 │  │ Action: BUY │  │ Action: HOLD│  │ Action: SELL│  │ Action: BUY │       │
 │  │ Conf: 0.8   │  │ Conf: 0.6   │  │ Conf: 0.9   │  │ Conf: 0.7   │       │
 │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
 │         │                 │                 │                 │           │
 │         └─────────────────┼─────────────────┼─────────────────┘           │
 │                           ▼                 ▼                             │
 │  ┌─────────────────────────────────────────────────────────────────────┐   │
 │  │                Consensus Signal                                     │   │
 │  │                                                                     │   │
 │  │  Weighted Average: Action = BUY, Confidence = 0.75                 │   │
 │  └─────────────────────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                 Reinforcement Learning Agent                               │
 │                                                                             │
 │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
 │  │   Neural    │    │  Enhanced   │    │   Final     │                     │
 │  │  Network    │───►│   State     │───►│  Decision   │                     │
 │  │ Processing  │    │ (with AI    │    │             │                     │
 │  │   (DQN)     │    │  signals)   │    │ Action: BUY │                     │
 │  └─────────────┘    └─────────────┘    │ Conf: 0.82  │                     │
 │         │                              │ Size: 3     │                     │
 │         │                              └─────────────┘                     │
 │         ▼                                      │                           │
 │  ┌─────────────┐                              │                           │
 │  │ Experience  │                              │                           │
 │  │   Buffer    │                              │                           │
 │  │ (Training)  │                              │                           │
 │  └─────────────┘                              │                           │
 └─────────────────────────────────────────────────┼─────────────────────────┘
                                                   │
                                                   ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      Trading Execution                                     │
 │                                                                             │
 │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
 │  │   Position  │    │    Risk     │    │   Order     │                     │
 │  │    Size     │───►│ Management  │───►│ Execution   │                     │
 │  │ Calculation │    │   Check     │    │             │                     │
 │  └─────────────┘    └─────────────┘    └─────────────┘                     │
 │         │                  │                  │                           │
 │         │                  │                  ▼                           │
 │         │                  │         ┌─────────────┐                       │
 │         │                  │         │ NinjaTrader │                       │
 │         │                  │         │   Signal    │                       │
 │         │                  │         │  (TCP Port  │                       │
 │         │                  │         │    5557)    │                       │
 │         │                  │         └─────────────┘                       │
 │         │                  │                  │                           │
 │         │                  │                  ▼                           │
 │         │                  │         ┌─────────────┐                       │
 │         │                  │         │   Trade     │                       │
 │         │                  │         │ Execution   │                       │
 │         │                  │         │   Result    │                       │
 │         │                  │         └─────────────┘                       │
 │         │                  │                  │                           │
 │         │                  │                  ▼                           │
 │         │                  │         ┌─────────────┐                       │
 │         │                  │         │  Feedback   │                       │
 │         │                  │         │    Loop     │                       │
 │         │                  │         │  (Reward    │                       │
 │         │                  │         │ Calculation)│                       │
 │         │                  │         └─────────────┘                       │
 │         │                  │                  │                           │
 │         └──────────────────┼──────────────────┘                           │
 │                            │                                               │
 │                            ▼                                               │
 │  ┌─────────────────────────────────────────────────────────────────────┐   │
 │  │                   Learning Update                                   │   │
 │  │                                                                     │   │
 │  │  • RL Agent Experience Storage                                      │   │
 │  │  • Neural Network Training                                          │   │
 │  │  • Subsystem Performance Update                                     │   │
 │  │  • Adaptive Architecture Optimization                               │   │
 │  └─────────────────────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NinjaTrader 8 with C# compilation enabled
- NVIDIA GPU recommended (CUDA support)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/dopamine-trading-system.git
   cd dopamine-trading-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup configuration**
   ```bash
   cp config/system_config.json.example config/system_config.json
   # Edit config/system_config.json with your settings
   ```

5. **Install NinjaTrader strategy**
   - Copy `ResearchStrategy.cs` to your NinjaTrader strategies folder
   - Compile the strategy in NinjaTrader
   - Apply the strategy to a chart

### Running the System

1. **Start NinjaTrader** and ensure the ResearchStrategy is active

2. **Run the Dopamine Trading System**
   ```bash
   python main.py
   ```

3. **Monitor the system**
   ```bash
   # With debug logging
   python main.py --log-level DEBUG
   
   # Custom configuration
   python main.py --config my_config.json
   ```

## 🧬 AI Components

### 1. **DNA Subsystem**
- **Genetic Algorithm**: Evolves trading patterns over time
- **Pattern Recognition**: Identifies recurring market structures
- **Fitness Selection**: Rewards successful pattern matches
- **Mutation & Crossover**: Generates new pattern variations

### 2. **Temporal Subsystem**
- **Cycle Detection**: Identifies market cycles using autocorrelation
- **Pattern Matching**: Recognizes temporal sequences
- **Time-of-Day Effects**: Learns intraday patterns
- **Seasonal Analysis**: Captures weekly/monthly effects

### 3. **Immune Subsystem**
- **Anomaly Detection**: Statistical outlier identification
- **Risk Pattern Matching**: Recognizes dangerous market conditions
- **Adaptive Thresholds**: Self-adjusting risk parameters
- **Threat Memory**: Learns from past market stress events

### 4. **Microstructure Subsystem**
- **Regime Detection**: Identifies market states (trending, ranging, volatile)
- **Order Flow Analysis**: Analyzes price-volume relationships
- **Liquidity Assessment**: Measures market liquidity conditions
- **Efficiency Analysis**: Detects mean reversion vs momentum

### 5. **Dopamine Subsystem**
- **Reward Prediction**: Estimates expected rewards
- **Surprise Detection**: Identifies unexpected market events
- **Learning Enhancement**: Modulates learning rates based on dopamine levels
- **Prediction Error**: Calculates and uses prediction errors for learning

## 🎯 Performance Features

### Adaptive Neural Networks
- **Dynamic Architecture**: Networks grow/shrink based on performance
- **Layer Importance**: Tracks and prunes underutilized layers
- **Architecture Adaptation**: Automatic network optimization
- **Meta-Learning**: Learns how to learn more effectively

### Reinforcement Learning
- **Deep Q-Network (DQN)**: Primary decision-making algorithm
- **Experience Replay**: Learns from past experiences
- **Exploration vs Exploitation**: Balances learning and performance
- **Multi-Objective Rewards**: Considers profit, risk, and efficiency

### Risk Management
- **Position Sizing**: Dynamic position sizing based on confidence
- **Drawdown Protection**: Automatic risk reduction during losses
- **Volatility Adjustment**: Adapts to market volatility
- **Multi-Level Stops**: Hierarchical risk management

## 📊 Configuration

### System Configuration (`config/system_config.json`)

```json
{
  "system": {
    "environment": "production",
    "log_level": "INFO",
    "tcp_host": "localhost",
    "tcp_port_data": 5556,
    "tcp_port_signals": 5557
  },
  "agent": {
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "exploration_rate": 0.1,
    "batch_size": 32,
    "memory_size": 10000
  },
  "subsystems": {
    "dna": {
      "enabled": true,
      "weight": 0.2,
      "pattern_length": 10,
      "mutation_rate": 0.01
    },
    "temporal": {
      "enabled": true,
      "weight": 0.2,
      "cycle_lengths": [5, 15, 60, 240],
      "lookback_periods": 100
    },
    "immune": {
      "enabled": true,
      "weight": 0.2,
      "anomaly_threshold": 2.0,
      "adaptation_rate": 0.05
    },
    "microstructure": {
      "enabled": true,
      "weight": 0.2,
      "regime_detection_window": 50,
      "volatility_threshold": 0.02
    },
    "dopamine": {
      "enabled": true,
      "weight": 0.2,
      "baseline_dopamine": 0.0,
      "surprise_threshold": 0.1
    }
  },
  "neural": {
    "hidden_layers": [256, 128, 64],
    "activation": "relu",
    "dropout_rate": 0.2,
    "learning_rate": 0.001
  }
}
```

## 📈 Performance Metrics

### System Targets
- **Startup Time**: < 10 seconds
- **Decision Latency**: < 100ms
- **Memory Usage**: < 2GB
- **CPU Usage**: < 50% single core

### Trading Metrics
- **Sharpe Ratio**: Optimized for risk-adjusted returns
- **Maximum Drawdown**: Controlled through immune system
- **Win Rate**: Enhanced through ensemble learning
- **Profit Factor**: Maximized through multi-objective optimization

## 🔧 Development

### Code Structure
```
dopamine_trading/
├── main.py                      # Entry point
├── config/                      # Configuration files
├── src/
│   ├── core/                   # Core system (TCP, data processing)
│   ├── agent/                  # RL agent implementation
│   ├── intelligence/           # AI subsystems
│   ├── neural/                 # Neural network components
│   └── shared/                 # Shared utilities and types
└── ResearchStrategy.cs         # NinjaTrader integration
```

### Adding New Subsystems

1. **Create subsystem class** implementing `AISubsystem` protocol
2. **Add to subsystem manager** in `src/intelligence/subsystem_manager.py`
3. **Update configuration** in `config/system_config.json`
4. **Register in trading system** initialization

### Neural Network Extensions

1. **Implement network class** inheriting from `nn.Module`
2. **Register with NetworkManager** in system initialization
3. **Add specialized training logic** if needed
4. **Update configuration** parameters

## 🛠️ Troubleshooting

### Common Issues

1. **TCP Connection Fails**
   - Check NinjaTrader is running
   - Verify TCP ports (5556, 5557) are correct
   - Ensure ResearchStrategy is active

2. **Import Errors**
   - Verify virtual environment is activated
   - Check all dependencies are installed
   - Ensure Python 3.8+ is being used

3. **Memory Issues**
   - Reduce neural network sizes in config
   - Decrease memory_size in agent config
   - Enable GPU if available

4. **Performance Issues**
   - Enable GPU acceleration
   - Reduce subsystem complexity
   - Optimize neural network architectures

### Debug Mode

```bash
# Enable debug logging
python main.py --log-level DEBUG

# Check system status
tail -f logs/trading_system.log
```

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)**: Detailed technical documentation
- **[Architecture Guide](docs/architecture.md)**: Deep dive into system design
- **[Configuration Reference](docs/configuration.md)**: Complete config options
- **[API Reference](docs/api.md)**: Code documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This trading system is for educational and research purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. Use at your own risk.

## 🌟 Acknowledgments

- Inspired by dopamine-based learning in neuroscience
- Built with PyTorch and modern deep learning techniques
- Integrates with NinjaTrader for professional trading
- Follows clean architecture principles for maintainability

---

<div align="center">

**Built with ❤️ and 🧠 by the Dopamine Trading Team**

[![GitHub stars](https://img.shields.io/github/stars/your-username/dopamine-trading-system.svg?style=social)](https://github.com/your-username/dopamine-trading-system)
[![GitHub forks](https://img.shields.io/github/forks/your-username/dopamine-trading-system.svg?style=social)](https://github.com/your-username/dopamine-trading-system)

</div>