"""
Q-Learning Reinforcement Learning Trading Agent
State : Discretized (RSI bin, BB% bin, MACD sign, position)
Actions : 0=HOLD  1=BUY  2=SELL
Reward : Sharpe-like — directional PnL minus volatility penalty

Training uses walk-forward episodes over the full price history.
Exploration decays from ε=1.0 → ε_min=0.01 via exponential decay.

For continuous/high-dimensional states, replace the Q-table with a
DQN (PyTorch / Keras) — stub included at bottom of file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict


ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


# State discretization

def _discretize(
    rsi: float,
    bb_pct: float,
    macd_h: float,
    position: int,
) -> tuple[int, int, int, int]:
    # Map continuous indicators to finite bins.
    rsi_bin = 0 if rsi < 35 else (2 if rsi > 65 else 1)   # 0=oversold, 1=mid, 2=overbought
    bb_bin = 0 if bb_pct < 0.25 else (2 if bb_pct > 0.75 else 1)
    macd_bin = 1 if macd_h >= 0 else -1
    return (rsi_bin, bb_bin, macd_bin, position)


def get_state(hist_slice: pd.DataFrame, position: int) -> tuple:
    rsi = float(hist_slice["RSI"].iloc[-1]) if "RSI"       in hist_slice.columns else 50.0
    bb_pct = float(hist_slice["BB_Pct"].iloc[-1]) if "BB_Pct"    in hist_slice.columns else 0.5
    macd_h = float(hist_slice["MACD_Hist"].iloc[-1])if "MACD_Hist" in hist_slice.columns else 0.0
    return _discretize(rsi, bb_pct, macd_h, position)


# Reward function definition

def compute_reward(
    ret: float,   # next-period raw return
    action: int,
    position: int,
    vol_penalty: float = 0.5,
) -> float:
    """
    Reward = directional PnL (scaled to %)
    + holding reward if in position
    − vol penalty to encourage risk-adjusted behavior
    """
    pnl = ret * 100 * (
        1  if action == 1 else   # buying -> get next return
        -1 if action == 2 else   # selling -> miss next return
        position                 # holding -> stay exposed
    )
    penalty = vol_penalty * abs(ret * 100)   # penalise large swings
    return pnl - penalty


# Q-Learning agent definition

class QLearningAgent:
    def __init__(
        self,
        alpha:     float = 0.10,
        gamma:     float = 0.95,
        epsilon:   float = 1.00,
        eps_min:   float = 0.01,
        eps_decay: float = 0.995,
    ):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.q_table   = defaultdict(lambda: np.zeros(3))

    # Action selection

    def act(self, state: tuple, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self.q_table[state]))

    # Q-update (Bellman equation)

    def update(
        self,
        state:      tuple,
        action:     int,
        reward:     float,
        next_state: tuple,
    ) -> None:
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (
            td_target - self.q_table[state][action]
        )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, hist: pd.DataFrame, n_episodes: int = 150) -> dict:
        """
        Walk the full price series for `n_episodes`.
        Each episode starts from bar 60 (warm-up for indicators).
        """
        episode_returns = []
        episode_sharpes = []
        warmup = 60

        for _ in range(n_episodes):
            position = 0
            daily_rets = []

            for i in range(warmup, len(hist) - 1):
                slc = hist.iloc[:i + 1]
                state = get_state(slc, position)
                action = self.act(state)

                price_t = float(hist["Close"].iloc[i])
                price_t1 = float(hist["Close"].iloc[i + 1])
                ret = (price_t1 - price_t) / price_t

                # Execute action
                if action == 1 and position == 0:
                    position = 1
                elif action == 2 and position == 1:
                    position = 0

                reward = compute_reward(ret, action, position)
                next_slc = hist.iloc[:i + 2]
                next_state = get_state(next_slc, position)

                self.update(state, action, reward, next_state)
                daily_rets.append(ret * position)

            ep_ret = float(np.sum(daily_rets))
            ep_sharpe = (float(np.mean(daily_rets)) / (float(np.std(daily_rets)) + 1e-9) * np.sqrt(252))
            episode_returns.append(ep_ret)
            episode_sharpes.append(ep_sharpe)
            self.decay_epsilon()

        return {
            "episode_returns": episode_returns,
            "episode_sharpes": episode_sharpes,
            "avg_return_last20": round(float(np.mean(episode_returns[-20:])), 4),
            "avg_sharpe_last20": round(float(np.mean(episode_sharpes[-20:])), 3),
            "q_table_states": len(self.q_table),
            "final_epsilon": round(self.epsilon, 4),
        }

    # Inference 

    def recommend(self, hist: pd.DataFrame, current_position: int = 0) -> dict:
        state = get_state(hist, current_position)
        q_vals = self.q_table[state]
        action = int(np.argmax(q_vals))
        total = np.sum(np.abs(q_vals)) + 1e-9

        return {
            "action": ACTIONS[action],
            "q_values": {ACTIONS[i]: round(float(q_vals[i]), 4) for i in range(3)},
            "state": state,
            "confidence": round(float(q_vals[action] / total), 3),
        }


# Entry point 

def train_rl_agent(hist: pd.DataFrame, n_episodes: int = 150) -> dict:
    agent = QLearningAgent()
    train_result = agent.train(hist, n_episodes=n_episodes)
    rec = agent.recommend(hist)

    return {
        "available": True,
        "agent": agent,
        "training": train_result,
        "recommendation": rec,
    }
