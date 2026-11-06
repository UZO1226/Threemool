"""자체 PPO Lag의 실제 설정. 이전 실행의 기본값을 그대로 명시한다."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 2_000_000
    n_env: int = 8
    horizon: int = 1148
    n_chunks: int = 25
    test_episodes: int = 10
    hidden_dim: int = 64
    init_log_std: float = -0.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    minibatch_size: int = 512
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    cost_limit: float = 15.0
    max_lambda: float = 20.0
    pid_kp: float = 0.02
    pid_ki: float = 0.005
    pid_kd: float = 0.01

    def __post_init__(self):
        ints = ("total_steps", "n_env", "horizon", "n_chunks", "test_episodes", "hidden_dim",
                "update_epochs", "minibatch_size")
        for name in ints:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        for name in ("actor_lr", "critic_lr", "target_kl", "max_grad_norm", "pid_ki"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("cost_limit", "max_lambda", "pid_kp", "pid_kd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip_ratio < 1:
            raise ValueError("Invalid gamma, gae_lambda, or clip_ratio")
        if self.n_chunks > self.n_updates:
            raise ValueError("n_chunks cannot exceed the number of rollout updates")

    @property
    def n_updates(self):
        return math.ceil(self.total_steps / (self.n_env * self.horizon))

    @property
    def actual_steps(self):
        return self.n_updates * self.n_env * self.horizon

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("PPO config must be a flat mapping; see ppo_wwtp.yaml")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown PPO settings: {sorted(unknown)}; see ppo_wwtp.yaml")
        return cls(**values)
