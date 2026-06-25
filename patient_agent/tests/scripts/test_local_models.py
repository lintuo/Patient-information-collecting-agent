#!/usr/bin/env python3
"""本地模型接入验证脚本 — 测试 ASR 和 Vision 链路是否正确注册。

用法：
    python test_local_models.py asr      # 测试 ASR
    python test_local_models.py vision    # 测试 Vision
    python test_local_models.py all       # 全部测试
"""
import sys

def test_factory_asr():
    print("\n=== ASR Factory 链路测试 ===")
    import os
    os.environ["PATIENT_AGENT_ASR_BACKEND"] = "local_asr"
    os.environ["PATIENT_AGENT_LOCAL_ASR_MODEL"] = "/home/amd-5e046r4/Project/models/Qwen3-ASR-1.7B"
    os.environ["PATIENT_AGENT_LOCAL_ASR_DEVICE"] = "cuda:0"

    from patient_agent.services.model_runtime.factory import (
        get_audio_client,
        get_runtime_config,
        BACKEND_LOCAL_ASR,
    )
    from patient_agent.services.model_runtime.transformers_asr import TransformersASRClient

    # 1. 检查常量存在
    print(f"  [OK] BACKEND_LOCAL_ASR = '{BACKEND_LOCAL_ASR}'")

    # 2. 检查 factory 能返回 local_asr
    client = get_audio_client(backend="local_asr")
    assert isinstance(client, TransformersASRClient), f"Expected TransformersASRClient, got {type(client)}"
    print(f"  [OK] get_audio_client(backend='local_asr') → {type(client).__name__}")

    # 3. 检查环境变量读取
    config = get_runtime_config()
    print(f"  [OK] asr_backend config = '{config['asr_backend']}'")
    print(f"  [OK] local_asr_model  = '{config['local_asr_model']}'")

    # 4. health_check（不加载模型，只检查路径存在）
    health = client.health_check()
    print(f"  [OK] health_check → {health}")
    print("\n✅ ASR 链路注册成功（模型加载需实际音频文件触发）")


def test_factory_vision():
    print("\n=== Vision Factory 链路测试 ===")
    import os
    os.environ["PATIENT_AGENT_VISION_PROVIDER"] = "transformers"
    os.environ["PATIENT_AGENT_VISION_HF_MODEL"] = "/home/amd-5e046r4/Project/models/Qwen3.5-4B"

    from patient_agent.services.vision.factory import (
        VisionService,
        TransformersVisionBackend,
        get_vision_service,
        rebuild_vision_service,
    )

    # 1. 服务能正确初始化
    service = rebuild_vision_service()
    assert isinstance(service.backend, TransformersVisionBackend), \
        f"Expected TransformersVisionBackend, got {type(service.backend)}"
    print(f"  [OK] VisionService(provider=transformers) → {type(service.backend).__name__}")
    print(f"  [OK] hf_model = '{service.config.hf_model}'")

    # 2. 单例模式
    s2 = get_vision_service()
    assert s2 is service, "get_vision_service() should return the same singleton"
    print(f"  [OK] get_vision_service() → singleton")

    # 3. 模拟文件不存在场景（检查错误路径）
    result = service.analyze("/nonexistent/image.jpg", "test-job", "test-file")
    assert result.success is False, "Should fail for nonexistent file"
    assert "not found" in (result.error or "").lower(), f"Error message: {result.error}"
    print(f"  [OK] nonexistent file → {result.error}")

    print("\n✅ Vision 链路注册成功（模型加载需实际图片触发）")


def test_vision_api_mode():
    print("\n=== Vision API 模式（默认）测试 ===")
    import os
    os.environ.pop("PATIENT_AGENT_VISION_PROVIDER", None)

    from patient_agent.services.vision.factory import VisionService, ApiVisionBackend

    service = VisionService()
    assert isinstance(service.backend, ApiVisionBackend), \
        f"Expected ApiVisionBackend (default), got {type(service.backend)}"
    print(f"  [OK] VisionService(default) → {type(service.backend).__name__}")
    print(f"  [OK] model = '{service.config.model}'")
    print(f"  [OK] base_url = '{service.config.base_url}'")
    print("\n✅ Vision API 模式正常（默认后端）")


def test_full_factory():
    print("\n=== 完整 factory 链路测试 ===")
    import os
    os.environ["PATIENT_AGENT_ASR_BACKEND"] = "local_asr"
    os.environ["PATIENT_AGENT_VISION_PROVIDER"] = "api"

    from patient_agent.services.model_runtime.factory import (
        get_model_runtime_client,
        get_chat_client,
        get_audio_client,
        clear_clients,
    )
    from patient_agent.services.model_runtime.mock_client import MockModelRuntimeClient
    from patient_agent.services.model_runtime.transformers_asr import TransformersASRClient

    clear_clients()

    # chat 默认走 mock
    chat = get_chat_client()
    print(f"  [OK] get_chat_client() → {type(chat).__name__}")

    # audio 走 local_asr
    audio = get_audio_client()
    assert isinstance(audio, TransformersASRClient)
    print(f"  [OK] get_audio_client() → {type(audio).__name__}")

    print("\n✅ 完整 factory 链路正确，各后端互不影响")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "asr":    [test_factory_asr, test_full_factory],
        "vision": [test_factory_vision, test_vision_api_mode],
        "all":    [test_factory_asr, test_full_factory,
                   test_factory_vision, test_vision_api_mode],
    }

    selected = tests.get(mode, tests["all"])
    passed = 0
    failed = 0

    for t in selected:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"\n❌ {t.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果: {passed} 通过, {failed} 失败")
    sys.exit(0 if failed == 0 else 1)
