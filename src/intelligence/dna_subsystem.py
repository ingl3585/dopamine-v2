"""
DNA Subsystem for pattern recognition and sequence analysis.
Implements genetic algorithm-inspired pattern evolution and market DNA detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import asyncio
import structlog

from ..shared.types import State, AISignal, SignalType, ActionType
from ..shared.constants import (
    DNA_DEFAULT_PATTERN_LENGTH, DNA_DEFAULT_MUTATION_RATE, DNA_MAX_PATTERNS
)

logger = structlog.get_logger(__name__)


class MarketPattern:
    """Represents a market pattern with genetic algorithm properties."""
    
    def __init__(self, sequence: np.ndarray, action: ActionType, fitness: float = 0.0):
        """Initialize market pattern.
        
        Args:
            sequence: Price/volume sequence pattern
            action: Associated trading action
            fitness: Pattern fitness score
        """
        self.sequence = sequence
        self.action = action
        self.fitness = fitness
        self.age = 0
        self.matches = 0
        self.successes = 0
        self.created_at = 0
    
    def calculate_fitness(self, success_rate: float, recency_bonus: float) -> float:
        """Calculate pattern fitness based on performance.
        
        Args:
            success_rate: Pattern success rate
            recency_bonus: Bonus for recent patterns
            
        Returns:
            float: Updated fitness score
        """
        base_fitness = success_rate
        age_penalty = max(0, self.age * 0.01)  # Gradual aging penalty
        match_bonus = min(0.2, self.matches * 0.01)  # Bonus for frequent matches
        
        self.fitness = base_fitness + recency_bonus + match_bonus - age_penalty
        return self.fitness
    
    def mutate(self, mutation_rate: float) -> 'MarketPattern':
        """Create mutated version of pattern.
        
        Args:
            mutation_rate: Probability of mutation per element
            
        Returns:
            MarketPattern: Mutated pattern
        """
        mutated_sequence = self.sequence.copy()
        
        # Apply random mutations
        for i in range(len(mutated_sequence)):
            if np.random.random() < mutation_rate:
                # Small random perturbation
                noise = np.random.normal(0, 0.1)
                mutated_sequence[i] += noise
        
        return MarketPattern(mutated_sequence, self.action, self.fitness * 0.8)


class DNASubsystem:
    """DNA-inspired pattern recognition subsystem."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DNA subsystem.
        
        Args:
            config: Subsystem configuration
        """
        self.config = config
        
        # Pattern parameters
        self.pattern_length = config.get("pattern_length", DNA_DEFAULT_PATTERN_LENGTH)
        self.mutation_rate = config.get("mutation_rate", DNA_DEFAULT_MUTATION_RATE)
        self.max_patterns = config.get("max_patterns", DNA_MAX_PATTERNS)
        
        # Pattern population
        self.patterns: List[MarketPattern] = []
        self.pattern_history = deque(maxlen=5000)
        
        # Pattern matching (reduced thresholds for easier activation)
        self.similarity_threshold = 0.70  # Reduced from 0.85
        self.min_matches_for_signal = 1   # Reduced from 3
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.pattern_performance: Dict[int, Dict[str, float]] = {}
        
        # Genetic algorithm parameters
        self.population_size = 50
        self.selection_pressure = 0.7
        self.crossover_rate = 0.3
        self.elite_size = 10
        
        logger.info(f"DNA subsystem initialized - pattern length: {self.pattern_length}, max patterns: {self.max_patterns}")
    
    async def analyze(self, state: State) -> Optional[AISignal]:
        """Analyze market state for DNA patterns.
        
        Args:
            state: Current market state
            
        Returns:
            Optional[AISignal]: DNA pattern signal or None
        """
        try:
            # Extract current pattern from state
            current_pattern = self._extract_pattern(state)
            if current_pattern is None:
                return None
            
            # Store pattern in history
            self.pattern_history.append(current_pattern)
            
            # Find matching patterns
            matches = self._find_matching_patterns(current_pattern)
            
            if len(matches) >= self.min_matches_for_signal:
                # Generate signal from best matches
                signal = self._generate_signal_from_matches(matches, state.timestamp)
                
                if signal:
                    # Evolve pattern population
                    await self._evolve_patterns()
                    
                    self.total_signals += 1
                    return signal
            
            # Periodically evolve patterns even without signals
            if len(self.pattern_history) % 100 == 0:
                await self._evolve_patterns()
            
            # Generate occasional test signal when enough patterns are collected
            if len(self.pattern_history) >= 10 and len(self.pattern_history) % 30 == 0:
                # Generate a low-confidence signal to stay active
                action = ActionType.HOLD  # Conservative default
                return AISignal(
                    signal_type=SignalType.DNA,
                    action=action,
                    confidence=0.4,  # Low confidence
                    strength=0.2,    # Low strength
                    metadata={"type": "pattern_test", "patterns_collected": len(self.pattern_history)},
                    timestamp=state.timestamp
                )
            
            return None
            
        except Exception as e:
            logger.error("DNA analysis failed", error=str(e))
            return None
    
    def _extract_pattern(self, state: State) -> Optional[np.ndarray]:
        """Extract pattern sequence from market state.
        
        Args:
            state: Market state
            
        Returns:
            Optional[np.ndarray]: Extracted pattern or None
        """
        # Use price momentum and volume features
        if len(state.prices) < self.pattern_length:
            return None
        
        # Combine price and volume information
        price_features = state.prices[-self.pattern_length:]
        volume_features = state.volumes[-self.pattern_length:] if len(state.volumes) >= self.pattern_length else []
        
        # Normalize features
        price_norm = (price_features - np.mean(price_features)) / (np.std(price_features) + 1e-8)
        
        if len(volume_features) > 0:
            volume_norm = (volume_features - np.mean(volume_features)) / (np.std(volume_features) + 1e-8)
            # Combine price and volume patterns
            pattern = np.concatenate([price_norm, volume_norm])
        else:
            pattern = price_norm
        
        return pattern
    
    def _find_matching_patterns(self, current_pattern: np.ndarray) -> List[MarketPattern]:
        """Find patterns similar to current pattern.
        
        Args:
            current_pattern: Current market pattern
            
        Returns:
            List[MarketPattern]: Matching patterns
        """
        matches = []
        
        for pattern in self.patterns:
            similarity = self._calculate_similarity(current_pattern, pattern.sequence)
            
            if similarity >= self.similarity_threshold:
                pattern.matches += 1
                matches.append(pattern)
        
        # Sort by fitness
        matches.sort(key=lambda p: p.fitness, reverse=True)
        
        return matches
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            float: Similarity score (0 to 1)
        """
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Use normalized correlation
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        
        # Handle NaN from constant patterns
        if np.isnan(correlation):
            return 0.0
        
        # Convert to 0-1 similarity score
        similarity = (correlation + 1.0) / 2.0
        
        return max(0.0, similarity)
    
    def _generate_signal_from_matches(self, matches: List[MarketPattern], 
                                    timestamp: int) -> Optional[AISignal]:
        """Generate trading signal from pattern matches.
        
        Args:
            matches: Matching patterns
            timestamp: Current timestamp
            
        Returns:
            Optional[AISignal]: Generated signal
        """
        if not matches:
            return None
        
        # Weight votes by pattern fitness
        action_votes = {ActionType.HOLD: 0.0, ActionType.BUY: 0.0, ActionType.SELL: 0.0}
        total_weight = 0.0
        
        for pattern in matches:
            weight = pattern.fitness
            action_votes[pattern.action] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Determine consensus action
        best_action = max(action_votes.items(), key=lambda x: x[1])[0]
        confidence = action_votes[best_action] / total_weight
        
        # Calculate signal strength
        strength = min(1.0, len(matches) / 10.0)  # Normalize by expected max matches
        
        # Add pattern-specific metadata
        metadata = {
            "pattern_matches": len(matches),
            "best_pattern_fitness": matches[0].fitness,
            "avg_pattern_age": np.mean([p.age for p in matches]),
            "pattern_diversity": len(set(p.action for p in matches))
        }
        
        return AISignal(
            signal_type=SignalType.DNA,
            action=best_action,
            confidence=confidence,
            strength=strength,
            metadata=metadata,
            timestamp=timestamp
        )
    
    async def _evolve_patterns(self) -> None:
        """Evolve pattern population using genetic algorithm."""
        try:
            if len(self.patterns) == 0:
                await self._initialize_population()
                return
            
            # Age all patterns
            for pattern in self.patterns:
                pattern.age += 1
            
            # Selection: keep elite patterns
            self.patterns.sort(key=lambda p: p.fitness, reverse=True)
            elite = self.patterns[:self.elite_size]
            
            # Generate new population
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Selection for breeding
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 if parent1.fitness > parent2.fitness else parent2
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = child.mutate(self.mutation_rate)
                
                new_population.append(child)
            
            # Replace population
            self.patterns = new_population[:self.max_patterns]
            
            # Update fitness based on recent performance
            self._update_pattern_fitness()
            
            logger.debug(
                "Pattern evolution complete",
                population_size=len(self.patterns),
                avg_fitness=np.mean([p.fitness for p in self.patterns]),
                elite_fitness=elite[0].fitness if elite else 0.0
            )
            
        except Exception as e:
            logger.error("Pattern evolution failed", error=str(e))
    
    async def _initialize_population(self) -> None:
        """Initialize pattern population from historical data."""
        if len(self.pattern_history) < self.pattern_length * 2:
            return
        
        # Extract random patterns from history
        history_array = np.array(list(self.pattern_history))
        
        for _ in range(self.population_size):
            # Random starting position
            start_idx = np.random.randint(0, len(history_array) - self.pattern_length)
            pattern_seq = history_array[start_idx:start_idx + self.pattern_length]
            
            # Random action assignment initially
            action = np.random.choice([ActionType.BUY, ActionType.SELL, ActionType.HOLD])
            
            pattern = MarketPattern(pattern_seq, action, np.random.random() * 0.5)
            self.patterns.append(pattern)
        
        logger.info("Pattern population initialized", size=len(self.patterns))
    
    def _tournament_selection(self) -> MarketPattern:
        """Select pattern using tournament selection.
        
        Returns:
            MarketPattern: Selected pattern
        """
        tournament_size = 3
        tournament = np.random.choice(self.patterns, tournament_size, replace=False)
        return max(tournament, key=lambda p: p.fitness)
    
    def _crossover(self, parent1: MarketPattern, parent2: MarketPattern) -> MarketPattern:
        """Create offspring through crossover.
        
        Args:
            parent1: First parent pattern
            parent2: Second parent pattern
            
        Returns:
            MarketPattern: Offspring pattern
        """
        # Single-point crossover
        crossover_point = np.random.randint(1, len(parent1.sequence))
        
        child_sequence = np.concatenate([
            parent1.sequence[:crossover_point],
            parent2.sequence[crossover_point:]
        ])
        
        # Inherit action from fitter parent
        child_action = parent1.action if parent1.fitness > parent2.fitness else parent2.action
        
        # Average fitness with some randomness
        child_fitness = (parent1.fitness + parent2.fitness) / 2.0 + np.random.normal(0, 0.1)
        
        return MarketPattern(child_sequence, child_action, max(0.0, child_fitness))
    
    def _update_pattern_fitness(self) -> None:
        """Update pattern fitness based on recent performance."""
        for pattern in self.patterns:
            # Calculate success rate
            success_rate = pattern.successes / max(pattern.matches, 1)
            
            # Recency bonus for newer patterns
            recency_bonus = max(0, (100 - pattern.age) / 1000.0)
            
            # Update fitness
            pattern.calculate_fitness(success_rate, recency_bonus)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get DNA subsystem performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.patterns:
            return {
                "total_signals": self.total_signals,
                "success_rate": 0.0,
                "pattern_count": 0,
                "avg_pattern_fitness": 0.0,
                "pattern_diversity": 0.0
            }
        
        success_rate = self.successful_signals / max(self.total_signals, 1)
        avg_fitness = np.mean([p.fitness for p in self.patterns])
        pattern_diversity = len(set(p.action for p in self.patterns)) / 3.0  # Normalize by max actions
        
        return {
            "total_signals": self.total_signals,
            "success_rate": success_rate,
            "pattern_count": len(self.patterns),
            "avg_pattern_fitness": avg_fitness,
            "pattern_diversity": pattern_diversity,
            "avg_pattern_age": np.mean([p.age for p in self.patterns]),
            "total_pattern_matches": sum(p.matches for p in self.patterns)
        }
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update subsystem parameters.
        
        Args:
            params: New parameter values
        """
        if "similarity_threshold" in params:
            self.similarity_threshold = params["similarity_threshold"]
        
        if "mutation_rate" in params:
            self.mutation_rate = params["mutation_rate"]
        
        if "min_matches_for_signal" in params:
            self.min_matches_for_signal = params["min_matches_for_signal"]
        
        logger.debug("DNA parameters updated", params=params)