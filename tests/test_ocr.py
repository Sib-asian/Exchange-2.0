"""Test per il modulo OCR di estrazione dati da screenshot."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.ocr import (
    ExtractedData,
    _check_gemini_available,
    _check_openai_available,
    _check_zai_available,
    _extract_with_gemini,
    _extract_with_openai,
    _extract_with_zai_cli,
    _fallback_extraction,
    _find_zai_command,
    _get_env_with_path,
    _get_gemini_api_key,
    _get_openai_api_key,
    _parse_vlm_response,
    _safe_float,
    extract_from_base64,
    extract_from_bytes,
    extract_from_image_file,
    validate_extracted_data,
)


class TestExtractedData:
    """Test per la dataclass ExtractedData."""

    def test_default_values(self):
        data = ExtractedData()
        assert data.squadra_casa == ""
        assert data.squadra_trasf == ""
        assert data.quota_1 == 0.0
        assert data.extraction_success is False
        assert data.backend_used == ""

    def test_to_dict(self):
        data = ExtractedData(
            squadra_casa="Juventus",
            squadra_trasf="Milan",
            quota_1=2.10,
            backend_used="gemini",
        )
        result = data.to_dict()
        assert result["squadra_casa"] == "Juventus"
        assert result["backend_used"] == "gemini"


class TestGetEnvWithPath:
    def test_includes_common_paths(self):
        env = _get_env_with_path()
        assert "PATH" in env
        assert "/usr/local/bin" in env["PATH"]


class TestFindZaiCommand:
    def test_returns_tuple(self):
        result = _find_zai_command()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCheckZaiAvailable:
    def test_returns_bool(self):
        result = _check_zai_available()
        assert isinstance(result, bool)


class TestGetGeminiApiKey:
    def test_from_environment(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        result = _get_gemini_api_key()
        assert result == "test-key"


class TestCheckGeminiAvailable:
    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        assert _check_gemini_available() is True


class TestGetOpenaiApiKey:
    def test_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        result = _get_openai_api_key()
        assert result == "test-openai-key"


class TestCheckOpenaiAvailable:
    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("src.ocr._get_openai_api_key", return_value=None):
            assert _check_openai_available() is False


class TestExtractWithZaiCli:
    def test_zai_not_available(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        with patch("src.ocr._find_zai_command", return_value=(None, None)):
            result = _extract_with_zai_cli(img_path)
            assert result.extraction_success is False

    def test_successful_extraction(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        mock_response = json.dumps({"squadra_casa": "Roma", "squadra_trasf": "Lazio", "quota_1": 1.85})

        with (
            patch("src.ocr._find_zai_command", return_value=("/usr/local/bin/z-ai", None)),
            patch("src.ocr.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout=mock_response, stderr="")
            result = _extract_with_zai_cli(img_path)
            assert result.extraction_success is True
            assert result.backend_used == "zai_cli"


class TestExtractWithGemini:
    def test_no_api_key(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        with patch("src.ocr._get_gemini_api_key", return_value=None):
            result = _extract_with_gemini(img_path)
            assert result.extraction_success is False

    def test_successful_extraction(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        mock_response = {"candidates": [{"content": {"parts": [{"text": '{"squadra_casa": "Inter", "quota_1": 2.0}'}]}}]}

        with (
            patch("src.ocr._get_gemini_api_key", return_value="test-key"),
            patch("src.ocr.urllib.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = _extract_with_gemini(img_path)
            assert result.extraction_success is True
            assert result.backend_used == "gemini"


class TestExtractWithOpenai:
    def test_no_api_key(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        with patch("src.ocr._get_openai_api_key", return_value=None):
            result = _extract_with_openai(img_path)
            assert result.extraction_success is False


class TestParseVlmResponse:
    def test_parse_valid_json(self):
        response = json.dumps({"squadra_casa": "Roma", "quota_1": 1.85})
        result = _parse_vlm_response(response)
        assert result.extraction_success is True
        assert result.squadra_casa == "Roma"

    def test_parse_json_with_markdown(self):
        response = '```json\n{"squadra_casa": "Inter"}\n```'
        result = _parse_vlm_response(response)
        assert result.extraction_success is True

    def test_parse_empty_response(self):
        result = _parse_vlm_response("")
        assert result.extraction_success is False


class TestFallbackExtraction:
    def test_extract_teams(self):
        result = _fallback_extraction("Roma vs Lazio", "test")
        assert "Roma" in result.squadra_casa


class TestSafeFloat:
    def test_float_conversion(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.50") == 2.50
        assert _safe_float("2,50") == 2.50
        assert _safe_float(None) == 0.0
        assert _safe_float("invalid") == 0.0


class TestValidateExtractedData:
    def test_valid_data(self):
        data = ExtractedData(quota_1=2.0, quota_x=3.3, quota_2=3.5)
        is_valid, warnings = validate_extracted_data(data)
        assert is_valid is True

    def test_missing_quotes(self):
        data = ExtractedData()
        is_valid, warnings = validate_extracted_data(data)
        assert is_valid is False


class TestExtractFromImageFile:
    def test_file_not_found(self):
        result = extract_from_image_file("/nonexistent.jpg")
        assert result.extraction_success is False

    def test_no_backend_available(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image")
        with (
            patch("src.ocr._check_zai_available", return_value=False),
            patch("src.ocr._check_gemini_available", return_value=False),
            patch("src.ocr._check_openai_available", return_value=False),
        ):
            result = extract_from_image_file(img_path)
            assert result.extraction_success is False
            assert "Nessun backend" in result.error_message


class TestExtractFromBytes:
    def test_creates_temp_file(self):
        with (
            patch("src.ocr._check_zai_available", return_value=True),
            patch("src.ocr._extract_with_zai_cli") as mock,
        ):
            mock.return_value = ExtractedData(extraction_success=True, backend_used="zai_cli")
            result = extract_from_bytes(b"fake", ".jpg")
            assert result.extraction_success is True


class TestExtractFromBase64:
    def test_valid_base64(self):
        import base64
        b64 = base64.b64encode(b"fake").decode()
        with (
            patch("src.ocr._check_zai_available", return_value=True),
            patch("src.ocr._extract_with_zai_cli") as mock,
        ):
            mock.return_value = ExtractedData(extraction_success=True)
            result = extract_from_base64(b64)
            assert result.extraction_success is True

    def test_invalid_base64(self):
        result = extract_from_base64("not valid!!!")
        assert result.extraction_success is False
