#!/usr/bin/env python3
"""
Main entry point for the Dopamine Trading System.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.trading_system import TradingSystem


async def main():
    """Main application entry point."""
    # Print startup banner
    print("=" * 60)
    print("🧠 Dopamine Trading System v2.0")
    print("AI-Powered Reinforcement Learning Trading")
    print("=" * 60)
    print()
    
    # Create and run trading system
    system = TradingSystem()
    
    try:
        await system.run()
    except KeyboardInterrupt:
        print("\n⚠️  Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    print("✅ System shutdown complete")
    return 0


if __name__ == "__main__":
    # Ensure we're in the correct directory
    os.chdir(Path(__file__).parent)
    
    # Run the main application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)