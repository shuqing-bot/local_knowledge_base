import json
import os
from pathlib import Path

# 获取项目根目录
BASE_DIR = Path(__file__).parent.absolute()


class Config:
    def __init__(self):
        self.config_path = BASE_DIR / "config.json"
        self._config = self._load_config()
        self._build_paths()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_paths(self):
        """自动拼接完整路径"""
        paths = self._config["paths"]

        # 自动生成完整的 CHECKPOINT_DB 路径
        paths["CHECKPOINT_DB"] = str(BASE_DIR / paths["CHECKPOINT_FILE"] / "chat_checkpoints.db")

        # 创建所有必要的目录
        for key in ["DATA_DIR", "PERSIST_DIR", "CHECKPOINT_FILE", "PROMPT_DIR"]:
            dir_path = BASE_DIR / paths[key]
            dir_path.mkdir(parents=True, exist_ok=True)

    def get(self, section: str, key: str = None):
        """获取配置"""
        if key is None:
            return self._config[section]
        return self._config[section][key]

    # 便捷属性
    @property
    def paths(self):
        return self._config["paths"]

    @property
    def models(self):
        return self._config["models"]

    @property
    def prompts(self):
        # 返回完整路径
        prompt_dir = BASE_DIR / self.paths["PROMPT_DIR"]
        return {k: str(prompt_dir / v) for k, v in self._config["prompts"].items()}

    @property
    def params(self):
        return self._config["parameters"]


