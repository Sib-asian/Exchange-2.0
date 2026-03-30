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

    def test_compact_keys_parsed(self):
        """Test che le chiavi compatte del prompt ridotto vengono parsate."""
        response = json.dumps({
            "min": 90, "g_h": 2, "g_a": 1,
            "r_h": 0, "r_a": 0, "y_h": 1, "y_a": 3,
            "sot_h": 4, "sot_a": 2, "soff_h": 7, "soff_a": 2,
            "blk_h": 4, "blk_a": 4,
            "cor_h": 6, "cor_a": 2,
            "pos_h": 50, "pos_a": 50,
            "att_h": 109, "att_a": 72,
            "datt_h": 48, "datt_a": 34,
            "fou_h": 18, "fou_a": 12,
        })
        result = _parse_live_stats_response(response)
        assert result.extraction_success is True
        assert result.minuto == 90
        assert result.gol_casa == 2
        assert result.gol_trasf == 1
        assert result.tiri_porta_casa == 4
        assert result.tiri_fuori_casa == 7
        assert result.corner_casa == 6
        assert result.possesso_casa == 50.0
        assert result.attacchi_pericolosi_casa == 48
        assert result.falli_casa == 18


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


# ---------------------------------------------------------------------------
# Tests for _parse_prematch_analysis_response
# ---------------------------------------------------------------------------

class TestParsePrematchAnalysisResponse:
    """Tests for the prematch analysis JSON parser."""

    def _make_full_json(self, **overrides) -> str:
        base = {
            "match": {
                "home_team": "Juventus",
                "away_team": "Inter",
                "league": "Serie A",
                "date": "2026-04-06",
            },
            "h2h": {
                "home_win_pct": 40, "draw_pct": 30, "away_win_pct": 30,
                "avg_goals_home": 1.2, "avg_goals_away": 0.9,
                "over_pct": 55, "ah_home_cover_pct": 48,
                "ht_home_win_pct": 35, "ht_draw_pct": 40, "ht_away_win_pct": 25,
            },
            "strength": {"home": 72, "away": 68},
            "odds": {"init_1": 2.10, "init_x": 3.40, "init_2": 3.20},
            "home": {
                "rank": 3, "matches": 28, "win": 18, "draw": 5, "lose": 5,
                "scored": 52, "conceded": 24, "win_rate": 68.0,
                "home_win": 10, "home_draw": 2, "home_lose": 2,
                "home_scored": 30, "home_conceded": 10,
                "last6_win": 4, "last6_draw": 1, "last6_lose": 1,
                "ht_win": 12, "ht_draw": 10, "ht_lose": 6,
                "goals_1h": 22, "goals_2h": 30,
            },
            "away": {
                "rank": 2, "matches": 28, "win": 17, "draw": 6, "lose": 5,
                "scored": 48, "conceded": 22, "win_rate": 64.0,
                "away_win": 7, "away_draw": 3, "away_lose": 4,
                "away_scored": 20, "away_conceded": 14,
                "last6_win": 3, "last6_draw": 2, "last6_lose": 1,
                "ht_win": 11, "ht_draw": 10, "ht_lose": 7,
                "goals_1h": 18, "goals_2h": 30,
            },
            "prev_home": {"win_pct": 65, "avg_scored": 1.9, "avg_conceded": 0.8, "over_pct": 60},
            "prev_away": {"win_pct": 55, "avg_scored": 1.6, "avg_conceded": 1.0, "over_pct": 50},
        }
        base.update(overrides)
        return json.dumps(base)

    def test_successful_parse(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.extraction_success is True
        assert result.home_team == "Juventus"
        assert result.away_team == "Inter"
        assert result.league_name == "Serie A"
        assert result.match_date == "2026-04-06"

    def test_h2h_fields(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.h2h_home_win_pct == 40.0
        assert result.h2h_draw_pct == 30.0
        assert result.h2h_away_win_pct == 30.0
        assert result.h2h_over_pct == 55.0
        assert result.h2h_ht_home_win_pct == 35.0

    def test_standings_home(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.home_rank == 3
        assert result.home_matches == 28
        assert result.home_win == 18
        assert result.home_scored == 52
        assert result.home_home_scored == 30.0

    def test_standings_away(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.away_rank == 2
        assert result.away_matches == 28
        assert result.away_away_scored == 20.0

    def test_market_odds(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.mkt_init_1 == 2.10
        assert result.mkt_init_x == 3.40
        assert result.mkt_init_2 == 3.20

    def test_forma_mult_calculated(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        # last6: 4W 1D 1L → strong form → forma_mult_h > 1.0
        assert result.forma_mult_h > 1.0
        # last6: 3W 2D 1L → decent form → forma_mult_a > 1.0
        assert result.forma_mult_a > 1.0

    def test_fixture_total_from_h2h(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        # fixture_total = blend(h2h_total=2.1, form_total) > 0
        assert result.fixture_historical_total > 0.5

    def test_prev_scores(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.home_prev_avg_scored == 1.9
        assert result.away_prev_avg_scored == 1.6

    def test_strength(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.strength_home == 72
        assert result.strength_away == 68

    def test_goal_timing(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response(self._make_full_json())
        assert result.home_goals_1h == 22.0
        assert result.away_goals_1h == 18.0

    def test_empty_response(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response("")
        assert result.extraction_success is False

    def test_invalid_json(self):
        from src.ocr import _parse_prematch_analysis_response
        result = _parse_prematch_analysis_response("not json")
        assert result.extraction_success is False

    def test_json_in_markdown(self):
        from src.ocr import _parse_prematch_analysis_response
        wrapped = f"```json\n{self._make_full_json()}\n```"
        result = _parse_prematch_analysis_response(wrapped)
        assert result.extraction_success is True
        assert result.home_team == "Juventus"

    def test_missing_match_section(self):
        """Parser should still work even without the match section."""
        from src.ocr import _parse_prematch_analysis_response
        data = json.loads(self._make_full_json())
        del data["match"]
        result = _parse_prematch_analysis_response(json.dumps(data))
        assert result.extraction_success is True
        assert result.home_team == ""
        assert result.away_team == ""

    def test_last6_autocorrect(self):
        """last6 values that don't sum to 6 should be auto-corrected."""
        from src.ocr import _parse_prematch_analysis_response
        data = json.loads(self._make_full_json())
        # Set last6 sum to 9 (wrong) — should be corrected to sum=6
        data["home"]["last6_win"] = 6
        data["home"]["last6_draw"] = 2
        data["home"]["last6_lose"] = 1
        result = _parse_prematch_analysis_response(json.dumps(data))
        assert result.extraction_success is True
        assert result.home_last6_win + result.home_last6_draw + result.home_last6_lose == 6


# ---------------------------------------------------------------------------
# Tests for session_storage
# ---------------------------------------------------------------------------

class TestSessionStorage:
    def test_load_empty(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        monkeypatch.setattr(ss, "_STORAGE_PATH", tmp_path / "test.json")
        assert ss.load_partite() == []

    def test_save_and_load(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        monkeypatch.setattr(ss, "_STORAGE_PATH", tmp_path / "test.json")
        p = ss.PartitaSalvata(id="abc", nome="Test", saved_at="01/01 10:00")
        ss.save_partita(p)
        loaded = ss.load_partite()
        assert len(loaded) == 1
        assert loaded[0].id == "abc"

    def test_update_existing(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        monkeypatch.setattr(ss, "_STORAGE_PATH", tmp_path / "test.json")
        p = ss.PartitaSalvata(id="abc", nome="Test", saved_at="01/01 10:00")
        ss.save_partita(p)
        p2 = ss.PartitaSalvata(id="abc", nome="Updated", saved_at="01/01 11:00")
        ss.save_partita(p2)
        loaded = ss.load_partite()
        assert len(loaded) == 1
        assert loaded[0].nome == "Updated"

    def test_max_partite(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        monkeypatch.setattr(ss, "_STORAGE_PATH", tmp_path / "test.json")
        monkeypatch.setattr(ss, "_MAX_PARTITE", 3)
        for i in range(5):
            ss.save_partita(ss.PartitaSalvata(id=str(i), nome=f"P{i}", saved_at="01/01"))
        loaded = ss.load_partite()
        assert len(loaded) == 3
        assert loaded[0].id == "2"  # oldest 0,1 removed

    def test_delete(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        monkeypatch.setattr(ss, "_STORAGE_PATH", tmp_path / "test.json")
        ss.save_partita(ss.PartitaSalvata(id="x", nome="X", saved_at="01/01"))
        ss.save_partita(ss.PartitaSalvata(id="y", nome="Y", saved_at="01/01"))
        ss.delete_partita("x")
        loaded = ss.load_partite()
        assert len(loaded) == 1
        assert loaded[0].id == "y"

    def test_build_partita_id(self):
        from src.session_storage import build_partita_id
        pid = build_partita_id()
        assert isinstance(pid, str)
        assert len(pid) > 0

    def test_build_saved_at_label(self):
        from src.session_storage import build_saved_at_label
        label = build_saved_at_label()
        assert "/" in label and ":" in label

    def test_collect_widget_state(self):
        from src.session_storage import collect_widget_state
        mock_state = {
            "lines_ah_op": -0.25,
            "bankroll_value": 1000.0,
            "non_serializable": object(),
        }
        result = collect_widget_state(mock_state)
        assert result["lines_ah_op"] == -0.25
        assert result["bankroll_value"] == 1000.0
        assert "non_serializable" not in result

    def test_restore_widget_state(self):
        from src.session_storage import restore_widget_state
        state = {}
        restore_widget_state(state, {"lines_ah_op": -0.5, "bankroll_value": 500.0})
        assert state["lines_ah_op"] == -0.5

    def test_corrupt_file(self, tmp_path, monkeypatch):
        import src.session_storage as ss
        p = tmp_path / "test.json"
        p.write_text("not valid json")
        monkeypatch.setattr(ss, "_STORAGE_PATH", p)
        result = ss.load_partite()
        assert result == []
