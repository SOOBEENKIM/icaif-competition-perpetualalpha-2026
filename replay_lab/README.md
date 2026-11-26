# 선택 전략의 공통 부품과 보존 결과

현재 개발 대상은 reversion_blend의 추세＋7일 되돌림 로직입니다. 기본 배율은 tail_calibration에서, 원래 배율 비교본은 reversion_blend에서 제공합니다. 실행 진입점은 프로젝트 루트의 strategy/입니다.

Python 코드는 실제 import 의존성만 남겼습니다. EMA·Donchian·Chronos·펀딩·원장이 연결되어 일부 예전 계열 이름이 남아 있습니다. 이름만 보고 폴더 전체를 삭제하면 선택 전략이 깨질 수 있습니다.

output/tail_calibration/20261014-v1 및 output/reversion_blend/20261014-v1의 원본 자료는 보존합니다. 동결 당시 소스 해시 목록은 당시의 다른 실험까지 넓게 포함합니다. 그 정확한 소스 182개는 ../cleanup_records/20261015_selected_strategy/selected_frozen_sources.zip에 보존했습니다. 과거 모든 실험을 복원하지 않고 python -m strategy.verify로 선택한 두 프로필을 검증할 수 있습니다.

동결 실행 패키지와 가중치는 output/tail_calibration/20261014-v1/runtime에 남아 있습니다. 앞으로 수정할 코드는 루트 strategy/와 공용 모듈이며, 원본 동결 패키지를 직접 수정하지 않습니다.

다른 실험의 원본 호가·체결·펀딩은 기존 경로에 유지했고 파생 결과·UI는 정리했습니다. ../cleanup_records/20261015_selected_strategy/manifest.json에 보존 사유·삭제 대상이 기록되어 있습니다. 평가 노출 이력은 output/evaluation_control에 유지합니다.
