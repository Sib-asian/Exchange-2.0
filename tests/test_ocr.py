"""Test per il modulo OCR di estrazione dati da screenshot."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.ocr import (
    ExtractedData,
    LiveStatsExtracted,
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
    _normalize_live_stats_keys,
    _parse_live_stats_response,
    _parse_vlm_response,
    _safe_float,
    _safe_int,
    extract_from_base64,
    extract_from_bytes,
    extract_from_image_file,
    extract_live_stats_from_bytes,
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


# ============================================================================
# Test per LiveStatsExtracted e funzioni correlate
# ============================================================================

class TestLiveStatsExtracted:
    def test_default_values(self):
        data = LiveStatsExtracted()
        assert data.minuto == 0
        assert data.gol_casa == 0
        assert data.corner_casa == 0
        assert data.possesso_casa == 0.0
        assert data.extraction_success is False

    def test_to_dict(self):
        data = LiveStatsExtracted(minuto=45, gol_casa=1, corner_casa=5)
        d = data.to_dict()
        assert d["minuto"] == 45
        assert d["gol_casa"] == 1
        assert d["corner_casa"] == 5


class TestSafeInt:
    def test_int_conversion(self):
        assert _safe_int(5) == 5
        assert _safe_int("3") == 3
        assert _safe_int(2.7) == 2
        assert _safe_int("4.9") == 4
        assert _safe_int(None) == 0
        assert _safe_int("abc") == 0


class TestNormalizeLiveStatsKeys:
    def test_italian_keys_unchanged(self):
        data = {"minuto": 45, "gol_casa": 1, "corner_casa": 5}
        result = _normalize_live_stats_keys(data)
        assert result["minuto"] == 45
        assert result["gol_casa"] == 1
        assert result["corner_casa"] == 5

    def test_english_keys_mapped(self):
        data = {
            "minute": 65,
            "goals_home": 2,
            "goals_away": 1,
            "shots_on_target_home": 7,
            "shots_on_target_away": 3,
            "shots_off_target_home": 5,
            "shots_off_target_away": 2,
            "corners_home": 6,
            "corners_away": 2,
            "possession_home": 58.0,
            "possession_away": 42.0,
            "dangerous_attacks_home": 45,
            "dangerous_attacks_away": 30,
            "fouls_home": 18,
            "fouls_away": 12,
            "yellow_cards_home": 1,
            "yellow_cards_away": 3,
            "red_cards_home": 0,
            "red_cards_away": 0,
        }
        result = _normalize_live_stats_keys(data)
        assert result["minuto"] == 65
        assert result["gol_casa"] == 2
        assert result["gol_trasf"] == 1
        assert result["tiri_porta_casa"] == 7
        assert result["tiri_fuori_casa"] == 5
        assert result["corner_casa"] == 6
        assert result["possesso_casa"] == 58.0
        assert result["attacchi_pericolosi_casa"] == 45
        assert result["falli_casa"] == 18
        assert result["gialli_casa"] == 1

    def test_nested_dict_flattened(self):
        """Test strutture annidate tipo {"shots_on_goal": {"home": 4, "away": 2}}."""
        data = {
            "minute": 90,
            "goals": {"home": 2, "away": 1},
            "corners": {"home": 6, "away": 2},
            "possession": {"home": 50, "away": 50},
            "shots_on_goal": {"home": 4, "away": 2},
            "dangerous_attacks": {"home": 48, "away": 34},
        }
        result = _normalize_live_stats_keys(data)
        assert result["minuto"] == 90
        assert result.get("gol_casa") == 2
        assert result.get("corner_casa") == 6
        assert result.get("tiri_porta_casa") == 4
        assert result.get("attacchi_pericolosi_casa") == 48

    def test_english_response_parsed_end_to_end(self):
        """Test che una risposta con chiavi inglesi viene parsata correttamente."""
        response = json.dumps({
            "minute": 90,
            "goals_home": 2,
            "goals_away": 1,
            "shots_on_goal_home": 4,
            "shots_on_goal_away": 2,
            "shots_off_goal_home": 7,
            "shots_off_goal_away": 2,
            "corners_home": 6,
            "corners_away": 2,
            "possession_home": 50,
            "possession_away": 50,
            "dangerous_attacks_home": 48,
            "dangerous_attacks_away": 34,
            "fouls_home": 18,
            "fouls_away": 12,
            "confidence": "high",
        })
        result = _parse_live_stats_response(response)
        assert result.extraction_success is True
        assert result.minuto == 90
        assert result.gol_casa == 2
        assert result.tiri_porta_casa == 4
        assert result.corner_casa == 6
        assert result.attacchi_pericolosi_casa == 48
        assert result.possesso_casa == 50.0


class TestParseLiveStatsResponse:
    def test_valid_json(self):
        response = json.dumps({
            "minuto": 65,
            "gol_casa": 2,
            "gol_trasf": 1,
            "rossi_casa": 0,
            "rossi_trasf": 1,
            "tiri_porta_casa": 7,
            "tiri_porta_trasf": 3,
            "tiri_fuori_casa": 5,
            "tiri_fuori_trasf": 4,
            "corner_casa": 6,
            "corner_trasf": 2,
            "possesso_casa": 58.0,
            "possesso_trasf": 42.0,
            "attacchi_pericolosi_casa": 45,
            "attacchi_pericolosi_trasf": 30,
            "confidence": "high",
        })
        result = _parse_live_stats_response(response)
        assert result.extraction_success is True
        assert result.minuto == 65
        assert result.gol_casa == 2
        assert result.corner_casa == 6
        assert result.possesso_casa == 58.0
        assert result.attacchi_pericolosi_casa == 45

    def test_json_in_markdown(self):
        response = '```json\n{"minuto": 30, "gol_casa": 0, "gol_trasf": 0}\n```'
        result = _parse_live_stats_response(response)
        assert result.extraction_success is True
        assert result.minuto == 30

    def test_empty_response(self):
        result = _parse_live_stats_response("")
        assert result.extraction_success is False

    def test_truncated_json_repair(self):
        # Simula JSON troncato (output tagliato a metà)
        response = '{\n    "minuto": 90,\n    "gol_casa": 2,\n    "gol_trasf": 1,\n    "corner_casa": 6,\n    "corner_tra'
        result = _parse_live_stats_response(response)
        assert result.extraction_success is True
        assert result.minuto == 90
        assert result.gol_casa == 2
        assert result.corner_casa == 6

    def test_invalid_json(self):
        result = _parse_live_stats_response("not json at all")
        assert result.extraction_success is False


class TestExtractLiveStatsFromBytes:
    def test_creates_temp_file(self):
        mock_response = {"candidates": [{"content": {"parts": [{"text": json.dumps({
            "minuto": 45, "gol_casa": 1, "gol_trasf": 0,
            "tiri_porta_casa": 5, "tiri_porta_trasf": 2,
            "corner_casa": 4, "corner_trasf": 1,
        })}]}}]}
        with (
            patch("src.ocr._get_gemini_api_key", return_value="test-key"),
            patch("src.ocr.urllib.request.urlopen") as mock_urlopen,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = extract_live_stats_from_bytes(b"fake image", ".jpg")
            assert result.extraction_success is True
            assert result.minuto == 45
            assert result.backend_used == "gemini"

    def test_no_api_key(self):
        with patch("src.ocr._get_gemini_api_key", return_value=None):
            result = extract_live_stats_from_bytes(b"fake image")
            assert result.extraction_success is False
            assert "API key" in result.error_message
