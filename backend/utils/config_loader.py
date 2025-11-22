"""
설정 파일 로더
training_presets.yaml을 읽어서 학습 파라미터 제공
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """학습 프리셋 설정 로더"""

    def __init__(self, config_path: str = "configs/training_presets.yaml"):
        self.config_path = Path(config_path)
        self._config = None

    def load(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        return self._config

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        특정 프리셋 가져오기

        Args:
            preset_name: 프리셋 이름 (test, character, style, concept 등)

        Returns:
            프리셋 설정 딕셔너리

        Raises:
            KeyError: 프리셋이 존재하지 않을 때
        """
        config = self.load()

        if preset_name not in config:
            available = list(config.keys())
            raise KeyError(
                f"프리셋 '{preset_name}'을 찾을 수 없습니다. "
                f"사용 가능: {available}"
            )

        # 기본값과 병합
        preset = config[preset_name].copy()
        if 'defaults' in config:
            # defaults를 먼저 적용하고 preset으로 덮어쓰기
            merged = config['defaults'].copy()
            merged.update(preset)
            preset = merged

        return preset

    def list_presets(self) -> list:
        """사용 가능한 프리셋 목록"""
        config = self.load()
        return [k for k in config.keys() if k != 'defaults']

    def get_preset_description(self, preset_name: str) -> str:
        """프리셋 설명 가져오기"""
        preset = self.get_preset(preset_name)
        return preset.get('description', 'No description')


# 사용 예시
if __name__ == "__main__":
    loader = ConfigLoader()

    # 모든 프리셋 출력
    print("사용 가능한 프리셋:")
    for name in loader.list_presets():
        desc = loader.get_preset_description(name)
        print(f"  - {name}: {desc}")

    # character 프리셋 로드
    print("\n캐릭터 프리셋 설정:")
    char_preset = loader.get_preset('character')
    for key, value in char_preset.items():
        print(f"  {key}: {value}")
