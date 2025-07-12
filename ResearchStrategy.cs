// ResearchStrategy.cs
// Advanced AI Trading Strategy with Multi-Entry Scaling and Position Reversal Capability
//
// Key Features:
// - Supports up to 10 entries per direction for position scaling
// - Automatic position reversals (long to short, short to long)
// - Each entry gets unique signal name for proper tracking
// - Individual stop loss and profit target management per entry
// - Enhanced position and execution tracking with safety limits
// - Real-time data streaming to Python AI system

using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using System.Globalization;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ResearchStrategy : Strategy
    {
        private TcpClient dataClient;
        private TcpClient signalClient;
        private Thread signalThread;
        private bool isConnected;
        private bool isRunning;
        private bool historicalDataSent = false;
        
        private List<double> prices1m = new List<double>();
        private List<double> prices5m = new List<double>();
        private List<double> prices15m = new List<double>();
        private List<double> prices1h = new List<double>();
        private List<double> prices4h = new List<double>();
        private List<double> volumes1m = new List<double>();
        private List<double> volumes5m = new List<double>();
        private List<double> volumes15m = new List<double>();
        private List<double> volumes1h = new List<double>();
        private List<double> volumes4h = new List<double>();
        
        // Enhanced account tracking
        private double lastAccountBalance = 0;
        private double lastBuyingPower = 0;
        private double sessionStartPnL = 0;
        private bool sessionStartSet = false;
		
		// Data sending tracking
        private DateTime lastDataSent = DateTime.MinValue;
        private int dataSendCount = 0;
        
        // Track entry counter for unique signal names when scaling in
        private int entryCounter = 0;
        private DateTime lastEntryTime = DateTime.MinValue;
        private int entriesThisMinute = 0;
		private int lastTradeCount = 0;
        
        protected override void OnStateChange()
        {
            switch (State)
            {
				case State.SetDefaults:
				    Description = "Adaptive AI Trading Strategy with Historical Bootstrapping";
				    Name = "ResearchStrategy";
				    Calculate = Calculate.OnBarClose;
				    BarsRequiredToTrade = 1;
				    
				    // CRITICAL: Allow multiple entries in same direction for scaling
				    EntriesPerDirection = 10;  // Allow up to 10 entries per direction
				    EntryHandling = EntryHandling.AllEntries;  // Process all entries until limit reached
				    break;
				
				case State.Configure:
				    AddDataSeries(BarsPeriodType.Minute, 15);  // BarsArray[1] - 15min
				    AddDataSeries(BarsPeriodType.Minute, 5);   // BarsArray[2] - 5min  
				    AddDataSeries(BarsPeriodType.Minute, 1);   // BarsArray[3] - 1min
				    AddDataSeries(BarsPeriodType.Minute, 60);  // BarsArray[4] - 1hour
				    AddDataSeries(BarsPeriodType.Minute, 240); // BarsArray[5] - 4hour
				    break;
                    
                case State.Realtime:
                    ConnectToPython();
                    StartSignalReceiver();
                    InitializeSession();
                    break;
                    
                case State.Terminated:
                    Cleanup();
                    break;
            }
        }
        
        private void InitializeSession()
        {
            if (!sessionStartSet)
            {
                sessionStartPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
                sessionStartSet = true;
                Print($"Session initialized - Starting P&L: {sessionStartPnL:C}");
            }
        }
        
		protected override void OnBarUpdate()
		{
		    // Only process on the primary series (BarsInProgress == 0)
		    // OnBarUpdate gets called for each timeframe: 0=primary, 1=15m, 2=5m, 3=1m
		    if (BarsInProgress != 0 || State != State.Realtime)
		        return;
		    
		    // Update price data from all available series when processing primary series
		    UpdatePriceData();
		    
		    if (!isConnected)
		        return;
		    
		    // Send historical data once
		    if (!historicalDataSent)
		    {
		        SendHistoricalData();
		        historicalDataSent = true;
		        return; // Don't send live data on the same bar as historical
		    }
		    
		    // Send live data on every bar close of primary series
		    if (HasValidData())
		    {
		        SendDataToPython();
		        dataSendCount++;		    
		    }
		}
		
		private bool HasValidData()
		{
		    // Check if we have valid data from primary series and at least some historical data
		    bool primaryValid = Close[0] > 0 && Volume[0] > 0;
		    bool listsHaveData = prices1m.Count > 0 && volumes1m.Count > 0;
		    
		    return primaryValid && listsHaveData;
		}

		private void SendHistoricalData()
		{
		    try
		    {
		        Print("Sending historical data to Python...");
		        
		        // Wait for all series to have some data before sending
		        if (BarsArray.Length < 3 || BarsArray[1].Count == 0 || BarsArray[2].Count == 0)
		        {
		            Print("Waiting for all timeframe data to load...");
		            return;
		        }
		        
		        // Get 10 days of data (approximately 1000+ bars for 15min)
		        int historyDays = 10;
		        int barsToSend15m = Math.Min(historyDays * 96, BarsArray[1].Count); // 96 15-min bars per day
		        int barsToSend5m = Math.Min(historyDays * 288, BarsArray[2].Count); // 288 5-min bars per day
		        int barsToSend1m = Math.Min(historyDays * 1440, BarsArray[3].Count); // 1440 1-min bars per day
		        int barsToSend1h = Math.Min(historyDays * 24, BarsArray[4].Count); // 24 1-hour bars per day
		        int barsToSend4h = Math.Min(historyDays * 6, BarsArray[5].Count); // 6 4-hour bars per day
		        
		        var historicalData = new
		        {
		            type = "historical_data",
		            bars_15m = GetHistoricalBars(BarsArray[1], barsToSend15m),
		            bars_5m = GetHistoricalBars(BarsArray[2], barsToSend5m),
		            bars_1m = GetHistoricalBars(BarsArray[3], barsToSend1m),
		            bars_1h = GetHistoricalBars(BarsArray[4], barsToSend1h),
		            bars_4h = GetHistoricalBars(BarsArray[5], barsToSend4h),
		            timestamp = DateTime.Now.Ticks
		        };
		        
		        string json = SerializeHistoricalData(historicalData);
		        SendJsonMessage(json);
		        
		        Print($"Historical data sent: {historicalData.bars_4h.Count} 4h bars, {historicalData.bars_1h.Count} 1h bars, {historicalData.bars_15m.Count} 15m bars, " +
		              $"{historicalData.bars_5m.Count} 5m bars, {historicalData.bars_1m.Count} 1m bars");
		              
		        historicalDataSent = true;
		    }
		    catch (Exception ex)
		    {
		        Print($"Historical data send error: {ex.Message}");
		        // Don't set historicalDataSent = true on error, so it will retry
		    }
		}
        
        private List<BarData> GetHistoricalBars(Bars bars, int count)
        {
            var barList = new List<BarData>();
            
            if (bars == null || bars.Count == 0)
                return barList;
            
            int startIndex = Math.Max(0, bars.Count - count);
            
            for (int i = startIndex; i < bars.Count; i++)
            {
                barList.Add(new BarData
                {
                    timestamp = bars.GetTime(i).Ticks,
                    open = bars.GetOpen(i),
                    high = bars.GetHigh(i),
                    low = bars.GetLow(i),
                    close = bars.GetClose(i),
                    volume = bars.GetVolume(i)
                });
            }
            
            return barList;
        }
        
        private string SerializeHistoricalData(dynamic data)
        {
            var sb = new StringBuilder();
            sb.Append("{");
            sb.Append($"\"type\":\"historical_data\",");
            sb.Append($"\"bars_15m\":{SerializeBarArray(data.bars_15m)},");
            sb.Append($"\"bars_5m\":{SerializeBarArray(data.bars_5m)},");
            sb.Append($"\"bars_1m\":{SerializeBarArray(data.bars_1m)},");
            sb.Append($"\"bars_1h\":{SerializeBarArray(data.bars_1h)},");
            sb.Append($"\"bars_4h\":{SerializeBarArray(data.bars_4h)},");
            sb.Append($"\"timestamp\":{data.timestamp}");
            sb.Append("}");
            return sb.ToString();
        }
        
        private string SerializeBarArray(List<BarData> bars)
        {
            var sb = new StringBuilder();
            sb.Append("[");
            
            for (int i = 0; i < bars.Count; i++)
            {
                if (i > 0) sb.Append(",");
                
                var bar = bars[i];
                sb.Append("{");
                sb.Append($"\"timestamp\":{bar.timestamp},");
                sb.Append($"\"open\":{bar.open.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"high\":{bar.high.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"low\":{bar.low.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"close\":{bar.close.ToString(CultureInfo.InvariantCulture)},");
                sb.Append($"\"volume\":{bar.volume}");
                sb.Append("}");
            }
            
            sb.Append("]");
            return sb.ToString();
        }

        private void UpdatePriceData()
        {
            // Only update when processing the primary series to avoid duplicate updates
            if (BarsInProgress != 0)
                return;

            // The primary series data (whatever timeframe the strategy is running on)
            // We'll treat this as our base timeframe
            UpdateList(prices1m, Close[0], 1000);
            UpdateList(volumes1m, Volume[0], 1000);

            // Update from additional series if they have data
            // BarsArray[1] = 15m series
            if (BarsArray.Length > 1 && BarsArray[1].Count > 0)
            {
                UpdateList(prices15m, Closes[1][0], 100);
                UpdateList(volumes15m, Volumes[1][0], 100);
            }

            // BarsArray[2] = 5m series  
            if (BarsArray.Length > 2 && BarsArray[2].Count > 0)
            {
                UpdateList(prices5m, Closes[2][0], 300);
                UpdateList(volumes5m, Volumes[2][0], 300);
            }

            // BarsArray[4] = 1h series
            if (BarsArray.Length > 4 && BarsArray[4].Count > 0)
            {
                UpdateList(prices1h, Closes[4][0], 50);
                UpdateList(volumes1h, Volumes[4][0], 50);
            }
            
            // BarsArray[5] = 4h series
            if (BarsArray.Length > 5 && BarsArray[5].Count > 0)
            {
                UpdateList(prices4h, Closes[5][0], 30);
                UpdateList(volumes4h, Volumes[5][0], 30);
            }
		    
		    // Note: BarsArray[3] would be 1m, but we're using the primary series as our minute data
            // If you want true 1m data separate from primary, you'd access it here as Closes[3][0]
        }
        
        private void UpdateList(List<double> list, double value, int maxSize)
        {
            list.Add(value);
            if (list.Count > maxSize)
                list.RemoveAt(0);
        }
        
		private void ConnectToPython()
		{
		    try
		    {
		        Print("Attempting to connect to Python AI system...");
		        
		        dataClient = new TcpClient("localhost", 5556);
		        signalClient = new TcpClient("localhost", 5557);
		        isConnected = true;
		        
		        Print("Connected to Python AI system successfully");
		    }
		    catch (Exception ex)
		    {
		        Print($"Connection failed: {ex.Message}");
		        isConnected = false;
		    }
		}
        
		private void SendDataToPython()
		{
		    if (!isConnected || dataClient?.Connected != true)
		    {
		        Print("Cannot send data - not connected to Python");
		        return;
		    }
		        
		    try
		    {
		        var json = BuildMarketDataJson();
		        
		        if (string.IsNullOrEmpty(json))
		        {
		            Print("Cannot send data - JSON is empty");
		            return;
		        }
		        
		        SendJsonMessage(json);
		    }
		    catch (Exception ex)
		    {
		        Print($"Data send error: {ex.Message}");
		        isConnected = false;
		    }
		}
        
        private void SendJsonMessage(string json)
        {
            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
            byte[] header = BitConverter.GetBytes(jsonBytes.Length);
            
            var stream = dataClient.GetStream();
            stream.Write(header, 0, 4);
            stream.Write(jsonBytes, 0, jsonBytes.Length);
        }
        
		private string BuildMarketDataJson()
		{
		    try
		    {
		        var sb = new StringBuilder();
		        sb.Append("{");
	
		        sb.Append($"\"type\":\"live_data\",");
		        sb.Append($"\"price_1m\":{SerializeDoubleArray(prices1m)},");
		        sb.Append($"\"price_5m\":{SerializeDoubleArray(prices5m)},");
                sb.Append($"\"price_15m\":{SerializeDoubleArray(prices15m)},");
                sb.Append($"\"price_1h\":{SerializeDoubleArray(prices1h)},");
                sb.Append($"\"price_4h\":{SerializeDoubleArray(prices4h)},");
		        sb.Append($"\"volume_1m\":{SerializeDoubleArray(volumes1m)},");
		        sb.Append($"\"volume_5m\":{SerializeDoubleArray(volumes5m)},");
		        sb.Append($"\"volume_15m\":{SerializeDoubleArray(volumes15m)},");
                sb.Append($"\"volume_1h\":{SerializeDoubleArray(volumes1h)},");
                sb.Append($"\"volume_4h\":{SerializeDoubleArray(volumes4h)},");
	
		        double currentBalance     = Account.Get(AccountItem.CashValue,             Currency.UsDollar);
		        double currentBuyingPower = Account.Get(AccountItem.ExcessIntradayMargin,  Currency.UsDollar);
		        double totalPnL           = Account.Get(AccountItem.RealizedProfitLoss,    Currency.UsDollar);
		        double unrealizedPnL      = Account.Get(AccountItem.UnrealizedProfitLoss,  Currency.UsDollar);
		        double dailyPnL           = sessionStartSet ? (totalPnL - sessionStartPnL) : 0;
		        double netLiquidation     = Account.Get(AccountItem.NetLiquidation,        Currency.UsDollar);
		        double marginUsed         = Account.Get(AccountItem.InitialMargin,         Currency.UsDollar);
		        double availableMargin    = currentBuyingPower;
	
		        if (currentBalance     <= 0) currentBalance     = 25000;
		        if (currentBuyingPower <= 0) currentBuyingPower = currentBalance;
		        if (netLiquidation     <= 0) netLiquidation     = currentBalance;
	
		        // Calculate enhanced market condition data for risk management
		        double currentVolatility = CalculateVolatility();
		        double drawdownPct = dailyPnL < 0 ? Math.Abs(dailyPnL / currentBalance) * 100.0 : 0.0;
		        double portfolioHeat = (marginUsed / netLiquidation) * 100.0;
		        int totalPositionSize = Math.Abs(Position.Quantity);
	
		        sb.Append($"\"account_balance\":{currentBalance.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"buying_power\":{currentBuyingPower.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"daily_pnl\":{dailyPnL.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"unrealized_pnl\":{unrealizedPnL.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"net_liquidation\":{netLiquidation.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"margin_used\":{marginUsed.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"available_margin\":{availableMargin.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"open_positions\":{Position.Quantity},");
		        sb.Append($"\"total_position_size\":{totalPositionSize},");
		        sb.Append($"\"current_price\":{Close[0].ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"volatility\":{currentVolatility.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"drawdown_pct\":{drawdownPct.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"portfolio_heat\":{portfolioHeat.ToString(CultureInfo.InvariantCulture)},");
		        sb.Append($"\"regime\":\"normal\",");
		        sb.Append($"\"trend_strength\":0.5,");
		        long marketTimeUnix = new DateTimeOffset(Time[0]).ToUnixTimeSeconds();
		        sb.Append($"\"timestamp\":{marketTimeUnix}");
	
		        sb.Append("}");
		        return sb.ToString();
		    }
		    catch (Exception ex)
		    {
		        Print($"Error building market data JSON: {ex.Message}");
		        return string.Empty;
		    }
		}
		
		private double CalculateVolatility()
		{
		    if (prices1m.Count < 20)
		        return 0.02; // Default 2% volatility
		        
		    double sum = 0;
		    int count = 0;
		    
		    for (int i = 1; i < Math.Min(20, prices1m.Count); i++)
		    {
		        if (prices1m[i-1] > 0)
		        {
		            double change = Math.Abs(prices1m[i] - prices1m[i-1]) / prices1m[i-1];
		            sum += change;
		            count++;
		        }
		    }
		    
		    return count > 0 ? sum / count : 0.02;
		}
        
        private string SerializeDoubleArray(List<double> array)
        {
            if (array.Count == 0) return "[]";
            
            var sb = new StringBuilder();
            sb.Append("[");
            
            for (int i = 0; i < array.Count; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(array[i].ToString(CultureInfo.InvariantCulture));
            }
            
            sb.Append("]");
            return sb.ToString();
        }
        
        private void StartSignalReceiver()
        {
            if (isRunning) return;
            
            isRunning = true;
            signalThread = new Thread(ReceiveSignals) { IsBackground = true };
            signalThread.Start();
        }
        
        private void ReceiveSignals()
        {
            while (isRunning)
            {
                try
                {
                    if (signalClient?.Connected != true)
                    {
                        Thread.Sleep(1000);
                        continue;
                    }
                    
                    byte[] header = new byte[4];
                    int bytesRead = signalClient.GetStream().Read(header, 0, 4);
                    if (bytesRead != 4) continue;
                    
                    int messageLength = BitConverter.ToInt32(header, 0);
                    if (messageLength <= 0 || messageLength > 10000) continue;
                    
                    byte[] messageBytes = new byte[messageLength];
                    bytesRead = signalClient.GetStream().Read(messageBytes, 0, messageLength);
                    if (bytesRead != messageLength) continue;
                    
                    string json = Encoding.UTF8.GetString(messageBytes);
                    ProcessSignal(json);
                }
                catch (Exception ex)
                {
                    Print($"Signal receive error: {ex.Message}");
                    Thread.Sleep(2000);
                }
            }
        }
        
        private void ProcessSignal(string json)
        {
            try
            {
                var signal = ParseSignalJson(json);
                
                int action = signal.Item1;
                double confidence = signal.Item2;
                int size = signal.Item3;
                bool useStop = signal.Item4;
                double stopPrice = signal.Item5;
                bool useTarget = signal.Item6;
                double targetPrice = signal.Item7;
                DateTime currentTime = DateTime.Now;
                
                // Check position size limits - but allow position reversals
                int currentPosition = Math.Abs(Position.Quantity);
                if (currentPosition >= 10)
                {
                    // Only block if we're trying to ADD to the same direction
                    if ((action == 1 && Position.MarketPosition == MarketPosition.Long) ||
                        (action == 2 && Position.MarketPosition == MarketPosition.Short))
                    {
                        Print($"Maximum position size reached ({currentPosition} contracts). Same-direction entry ignored.");
                        return;
                    }
                    // Allow reversals to proceed
                    Print($"Position reversal allowed: {Position.MarketPosition} {currentPosition} -> {(action == 1 ? "Long" : "Short")} {size}");
                }
                
                // Generate unique entry name for scaling capability
                entryCounter++;
                string entryName = $"AI_{DateTime.Now:HHmmss}_{entryCounter}";
                
                // Enhanced logic: Handle scaling in same direction vs position reversals
                switch (action)
                {
                    case 1: // Buy Signal
                        if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Scaling into existing long position
                            Print($"Scaling into LONG position: +{size} contracts (Current: {Position.Quantity})");
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Reversing from short to long - NinjaTrader handles this automatically
                            int totalSize = Math.Abs(Position.Quantity) + size;
                            Print($"REVERSING position: Short {Math.Abs(Position.Quantity)} -> Long {size} (Total Long: {size})");
                        }
                        else
                        {
                            // New long position from flat
                            Print($"New LONG position: {size} contracts");
                        }
                        
                        if (size > 0)
                        {
                            EnterLong(size, entryName);
                            entriesThisMinute++;
                            lastEntryTime = currentTime;
                        }
                        break;
                        
                    case 2: // Sell Signal
                        if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Scaling into existing short position
                            Print($"Scaling into SHORT position: +{size} contracts (Current: {Position.Quantity})");
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Reversing from long to short - NinjaTrader handles this automatically
                            int totalSize = Math.Abs(Position.Quantity) + size;
                            Print($"REVERSING position: Long {Math.Abs(Position.Quantity)} -> Short {size} (Total Short: {size})");
                        }
                        else
                        {
                            // New short position from flat
                            Print($"New SHORT position: {size} contracts");
                        }
                        
                        if (size > 0)
                        {
                            EnterShort(size, entryName);
                            entriesThisMinute++;
                            lastEntryTime = currentTime;
                        }
                        break;
                }
                
                // Set stop loss and profit target for this specific entry
                if (useStop && stopPrice > 0)
                {
                    SetStopLoss(entryName, CalculationMode.Price, stopPrice, false);
                    Print($"Stop set for {entryName} at {stopPrice:F2}");
                }
                    
                if (useTarget && targetPrice > 0)
                {
                    SetProfitTarget(entryName, CalculationMode.Price, targetPrice);
                    Print($"Target set for {entryName} at {targetPrice:F2}");
                }
                
                Print($"AI Signal Processed: {(action == 1 ? "BUY" : "SELL")} {size} contracts " +
                      $"(Entry: {entryName}, Conf: {confidence:P0}, New Position: {Position.Quantity})");
                
                // Reset counter periodically to prevent overflow
                if (entryCounter > 999)
                    entryCounter = 0;
                
            }
            catch (Exception ex)
            {
                Print($"Signal processing error: {ex.Message}");
            }
        }
        
        private Tuple<int, double, int, bool, double, bool, double> ParseSignalJson(string json)
        {
            int action = 0;
            double confidence = 0.0;
            int size = 1;
            bool useStop = false;
            double stopPrice = 0.0;
            bool useTarget = false;
            double targetPrice = 0.0;
            
            try
            {
                action = ExtractIntValue(json, "action");
                confidence = ExtractDoubleValue(json, "confidence");
                size = ExtractIntValue(json, "position_size");
                useStop = ExtractBoolValue(json, "use_stop");
                stopPrice = ExtractDoubleValue(json, "stop_price");
                useTarget = ExtractBoolValue(json, "use_target");
                targetPrice = ExtractDoubleValue(json, "target_price");
            }
            catch (Exception ex)
            {
                Print($"JSON parsing error: {ex.Message}");
            }
            
            return new Tuple<int, double, int, bool, double, bool, double>(
                action, confidence, size, useStop, stopPrice, useTarget, targetPrice);
        }
        
        private int ExtractIntValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return 0;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return 0;
            
            int startIndex = colonIndex + 1;
            while (startIndex < json.Length && (json[startIndex] == ' ' || json[startIndex] == '\t'))
                startIndex++;
            
            int endIndex = startIndex;
            while (endIndex < json.Length && char.IsDigit(json[endIndex]))
                endIndex++;
            
            if (endIndex > startIndex && int.TryParse(json.Substring(startIndex, endIndex - startIndex), out int result))
                return result;
            
            return 0;
        }
        
        private double ExtractDoubleValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return 0.0;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return 0.0;
            
            int startIndex = colonIndex + 1;
            while (startIndex < json.Length && (json[startIndex] == ' ' || json[startIndex] == '\t'))
                startIndex++;
            
            int endIndex = startIndex;
            while (endIndex < json.Length && (char.IsDigit(json[endIndex]) || json[endIndex] == '.' || json[endIndex] == '-'))
                endIndex++;
            
            if (endIndex > startIndex && double.TryParse(json.Substring(startIndex, endIndex - startIndex), NumberStyles.Float, CultureInfo.InvariantCulture, out double result))
                return result;
            
            return 0.0;
        }
        
        private bool ExtractBoolValue(string json, string key)
        {
            string pattern = $"\"{key}\"";
            int keyIndex = json.IndexOf(pattern);
            if (keyIndex == -1) return false;
            
            int colonIndex = json.IndexOf(":", keyIndex);
            if (colonIndex == -1) return false;
            
            int trueIndex = json.IndexOf("true", colonIndex);
            int falseIndex = json.IndexOf("false", colonIndex);
            
            if (trueIndex != -1 && (falseIndex == -1 || trueIndex < falseIndex))
                return true;
            
            return false;
        }
        
		protected override void OnExecutionUpdate(Execution execution, string executionId, 
		                                        double price, int quantity, MarketPosition marketPosition, 
		                                        string orderId, DateTime time)
		{
		    // Track completion of any AI-generated order (for scaling positions)
		    if (execution.Order?.Name?.Contains("AI_") == true)
		    {
		        Print($"Execution: {execution.Order.Name}, Price: {price:F2}, Qty: {quantity}, Position: {marketPosition}");
		        
		        // ENHANCED: Check for completed trades instead of just flat positions
		        CheckForCompletedTrades();
		    }
		}
		
		private void CheckForCompletedTrades()
		{
		    int currentTradeCount = SystemPerformance.AllTrades.Count;
		    
		    // If we have new completed trades
		    if (currentTradeCount > lastTradeCount)
		    {
		        // Send completion data for each new completed trade
		        for (int i = lastTradeCount; i < currentTradeCount; i++)
		        {
		            var completedTrade = SystemPerformance.AllTrades[i];
		            SendTradeCompletionForTrade(completedTrade);
		        }
		        
		        lastTradeCount = currentTradeCount;
		    }
		}
        
		private void SendTradeCompletionForTrade(Trade completedTrade)
		{
		    if (!isConnected || dataClient?.Connected != true)
		        return;
		        
		    try
		    {
		        double pnl = completedTrade.ProfitCurrency;
		        double entryPrice = completedTrade.Entry.Price;
		        double exitPrice = completedTrade.Exit.Price;
		        int quantity = completedTrade.Quantity;
		        DateTime entryTime = completedTrade.Entry.Time;
		        DateTime exitTime = completedTrade.Exit.Time;
		        
		        // Determine exit reason for risk learning
		        string exitReason = "manual_exit"; // Default
		        if (completedTrade.Exit.Name.Contains("Stop"))
		            exitReason = "stop_hit";
		        else if (completedTrade.Exit.Name.Contains("Target"))
		            exitReason = "target_hit";
		        else if (completedTrade.Exit.Name.Contains("AI_"))
		            exitReason = "ai_exit";
		        
		        // Enhanced trade completion data for risk learning
		        double currentBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);
		        double netLiquidation = Account.Get(AccountItem.NetLiquidation, Currency.UsDollar);
		        double marginUsed = Account.Get(AccountItem.InitialMargin, Currency.UsDollar);
		        double totalPnL = Account.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar);
		        double dailyPnL = sessionStartSet ? (totalPnL - sessionStartPnL) : 0;
		        
		        // Calculate trade duration and market conditions
		        double tradeDurationMinutes = (exitTime - entryTime).TotalMinutes;
		        double priceMove = Math.Abs(exitPrice - entryPrice);
		        double priceMovePct = (priceMove / entryPrice) * 100.0;
		        
		        // Estimate volatility from recent price action
		        double currentVolatility = 0.02; // Default 2%
		        if (prices1m.Count >= 20)
		        {
		            double sum = 0;
		            for (int i = 1; i < Math.Min(20, prices1m.Count); i++)
		            {
		                double change = Math.Abs(prices1m[i] - prices1m[i-1]) / prices1m[i-1];
		                sum += change;
		            }
		            currentVolatility = sum / Math.Min(19, prices1m.Count - 1);
		        }
		        
		        var json = $"{{" +
		                  $"\"type\":\"trade_completion\"," +
		                  $"\"pnl\":{pnl.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"exit_price\":{exitPrice.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"entry_price\":{entryPrice.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"size\":{quantity}," +
		                  $"\"exit_reason\":\"{exitReason}\"," +
		                  $"\"entry_time\":{entryTime.Ticks}," +
		                  $"\"exit_time\":{exitTime.Ticks}," +
		                  $"\"account_balance\":{currentBalance.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"net_liquidation\":{netLiquidation.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"margin_used\":{marginUsed.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"daily_pnl\":{dailyPnL.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"trade_duration_minutes\":{tradeDurationMinutes.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"price_move_pct\":{priceMovePct.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"volatility\":{currentVolatility.ToString(CultureInfo.InvariantCulture)}," +
		                  $"\"regime\":\"normal\"," +
		                  $"\"trend_strength\":0.5," +
		                  $"\"confidence\":0.5," +
		                  $"\"consensus_strength\":0.5," +
		                  $"\"primary_tool\":\"ai_signal\"," +
		                  $"\"timestamp\":{Time[0].Ticks}" +
		                  $"}}";
		        
		        SendJsonMessage(json);
		        Print($"Enhanced trade completion sent: P&L ${pnl:F2} ({exitReason}) - Size: {quantity}, Duration: {tradeDurationMinutes:F1}min, Vol: {currentVolatility:P2}");
		    }
		    catch (Exception ex)
		    {
		        Print($"Trade completion send error: {ex.Message}");
		    }
		}
        
        private void Cleanup()
        {
            isRunning = false;
            isConnected = false;
            
            try
            {
                dataClient?.Close();
                signalClient?.Close();
                signalThread?.Join(2000);
            }
            catch (Exception ex)
            {
                Print($"Cleanup error: {ex.Message}");
            }
        }
    }
    
    public class BarData
    {
        public long timestamp;
        public double open;
        public double high;
        public double low;
        public double close;
        public long volume;
    }
}