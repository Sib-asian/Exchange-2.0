"""Test per il modulo OCR di estrazione dati da screenshot."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ocr import (
    BUN_PATHS,
    ZAI_CLI_PATHS,
    ExtractedData,
    _fallback_extraction,
    _find_bun,
    _find_executable,
    _find_zai_cli,
    _get_subprocess_env,
    _parse_vlm_response,
    _run_vision_command,
    _safe_float,
    extract_from_base64,
    extract_from_bytes,
    extract_from_image_file,
    validate_extracted_data,
)


class TestExtractedData:
    """Test per la dataclass ExtractedData."""

    def test_default_values(self):
        """Verifica valori di default."""
        data = ExtractedData()
        assert data.squadra_casa == ""
        assert data.squadra_trasf == ""
        assert data.quota_1 == 0.0
        assert data.quota_x == 0.0
        assert data.quota_2 == 0.0
        assert data.linea_ou == 0.0
        assert data.quota_over == 0.0
        assert data.quota_under == 0.0
        assert data.quota_gg == 0.0
        assert data.quota_ng == 0.0
        assert data.extraction_success is False
        assert data.error_message == ""
        assert data.confidence == "medium"

    def test_to_dict(self):
        """Verifica conversione in dizionario."""
        data = ExtractedData(
            squadra_casa="Juventus",
            squadra_trasf="Milan",
            quota_1=2.10,
            quota_x=3.40,
            quota_2=3.20,
            extraction_success=True,
            confidence="high",
        )
        result = data.to_dict()
        assert result["squadra_casa"] == "Juventus"
        assert result["squadra_trasf"] == "Milan"
        assert result["quota_1"] == 2.10
        assert result["quota_x"] == 3.40
        assert result["quota_2"] == 3.20
        assert result["extraction_success"] is True
        assert result["confidence"] == "high"


class TestFindExecutable:
    """Test per le funzioni di ricerca eseguibili."""

    def test_find_executable_with_which(self):
        """Verifica che _find_executable trovi un eseguibile con shutil.which."""
        with patch("src.ocr.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/python3"
            result = _find_executable("python3", [])
            assert result == "/usr/bin/python3"

    def test_find_executable_with_paths(self):
        """Verifica ricerca in percorsi specifici."""
        with patch("src.ocr.shutil.which") as mock_which:
            mock_which.return_value = None
            with patch("src.ocr.os.path.isfile") as mock_isfile:
                with patch("src.ocr.os.access") as mock_access:
                    mock_isfile.return_value = True
                    mock_access.return_value = True
                    result = _find_executable("test", ["/fake/path/test"])
                    assert result == "/fake/path/test"

    def test_find_bun_returns_string(self):
        """Verifica che _find_bun ritorni una stringa."""
        result = _find_bun()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_find_zai_cli_returns_path_or_none(self):
        """Verifica che _find_zai_cli ritorni un path o None."""
        result = _find_zai_cli()
        assert result is None or isinstance(result, str)


class TestGetSubprocessEnv:
    """Test per _get_subprocess_env."""

    def test_includes_common_paths(self):
        """Verifica che i path comuni siano inclusi."""
        env = _get_subprocess_env()
        assert "PATH" in env
        assert "/usr/local/bin" in env["PATH"]
        assert "/usr/bin" in env["PATH"]


class TestParseVlmResponse:
    """Test per _parse_vlm_response."""

    def test_parse_valid_json(self):
        """Verifica parsing di JSON valido."""
        response = json.dumps({
            "squadra_casa": "Roma",
            "squadra_trasf": "Lazio",
            "quota_1": 1.85,
            "quota_x": 3.60,
            "quota_2": 4.20,
            "linea_ou": 2.5,
            "quota_over": 1.95,
            "quota_under": 1.90,
            "quota_gg": 1.75,
            "quota_ng": 2.05,
            "confidence": "high",
        })
        result = _parse_vlm_response(response)
        assert result.extraction_success is True
        assert result.squadra_casa == "Roma"
        assert result.squadra_trasf == "Lazio"
        assert result.quota_1 == 1.85
        assert result.confidence == "high"

    def test_parse_json_with_markdown(self):
        """Verifica parsing di JSON in markdown code block."""
        response = """```json
{
    "squadra_casa": "Inter",
    "squadra_trasf": "Napoli",
    "quota_1": 2.00,
    "quota_x": 3.30,
    "quota_2": 3.50,
    "confidence": "medium"
}
```"""
        result = _parse_vlm_response(response)
        assert result.extraction_success is True
        assert result.squadra_casa == "Inter"
        assert result.squadra_trasf == "Napoli"

    def test_parse_empty_response(self):
        """Verifica gestione risposta vuota."""
        result = _parse_vlm_response("")
        assert result.extraction_success is False
        assert "vuota" in result.error_message.lower()

    def test_parse_invalid_json_uses_fallback(self):
        """Verifica che JSON invalido attivi il fallback."""
        response = "Roma vs Lazio con quote 1.85, 3.60, 4.20"
        result = _parse_vlm_response(response)
        # Il fallback dovrebbe estrarre qualcosa
        assert result.raw_response == response


class TestFallbackExtraction:
    """Test per _fallback_extraction."""

    def test_extract_teams_with_vs(self):
        """Verifica estrazione squadre con 'vs'."""
        response = "Roma vs Lazio ha quote interessanti"
        result = _fallback_extraction(response, "test error")
        assert "Roma" in result.squadra_casa
        assert "Lazio" in result.squadra_trasf

    def test_extract_teams_with_dash(self):
        """Verifica estrazione squadre con '-'."""
        response = "Inter - Milan: le quote sono"
        result = _fallback_extraction(response, "test error")
        assert result.squadra_casa == "Inter"
        assert result.squadra_trasf == "Milan"

    def test_extract_quotes(self):
        """Verifica estrazione quote."""
        response = "Quote: 1.85, 3.60, 4.20 per questa partita"
        result = _fallback_extraction(response, "test error")
        # Le prime 3 quote dovrebbero essere assegnate a 1, X, 2
        assert result.quota_1 == 1.85
        assert result.quota_x == 3.60
        assert result.quota_2 == 4.20


class TestSafeFloat:
    """Test per _safe_float."""

    def test_float_to_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self):
        assert _safe_float(42) == 42.0

    def test_string_to_float(self):
        assert _safe_float("2.50") == 2.50

    def test_string_with_comma(self):
        assert _safe_float("2,50") == 2.50

    def test_none_returns_zero(self):
        assert _safe_float(None) == 0.0

    def test_invalid_returns_zero(self):
        assert _safe_float("invalid") == 0.0


class TestValidateExtractedData:
    """Test per validate_extracted_data."""

    def test_valid_complete_data(self):
        """Verifica dati completi e validi."""
        data = ExtractedData(
            squadra_casa="Juventus",
            squadra_trasf="Milan",
            quota_1=2.10,
            quota_x=3.40,
            quota_2=3.20,
            linea_ou=2.5,
            quota_over=1.95,
            quota_under=1.90,
            quota_gg=1.75,
            quota_ng=2.05,
            extraction_success=True,
        )
        is_valid, warnings = validate_extracted_data(data)
        assert is_valid is True
        assert len(warnings) == 0

    def test_missing_teams(self):
        """Verifica warning per squadre mancanti."""
        data = ExtractedData(
            quota_1=2.10,
            quota_x=3.40,
            quota_2=3.20,
        )
        is_valid, warnings = validate_extracted_data(data)
        assert is_valid is True  # Le quote 1X2 ci sono
        assert any("casa" in w.lower() for w in warnings)
        assert any("trasferta" in w.lower() for w in warnings)

    def test_missing_1x2_quotes(self):
        """Verifica non valido per quote 1X2 mancanti."""
        data = ExtractedData(
            squadra_casa="Juventus",
            squadra_trasf="Milan",
        )
        is_valid, warnings = validate_extracted_data(data)
        assert is_valid is False
        assert any("1X2" in w for w in warnings)

    def test_quote_out_of_range(self):
        """Verifica warning per quote fuori range."""
        data = ExtractedData(
            quota_1=0.50,  # Troppo bassa
            quota_x=3.40,
            quota_2=100.0,  # Troppo alta
        )
        is_valid, warnings = validate_extracted_data(data)
        assert any("fuori range" in w for w in warnings)


class TestExtractFromImageFile:
    """Test per extract_from_image_file."""

    def test_file_not_found(self):
        """Verifica gestione file inesistente."""
        result = extract_from_image_file("/nonexistent/path/image.jpg")
        assert result.extraction_success is False
        assert "non trovato" in result.error_message.lower()

    def test_successful_extraction(self, tmp_path):
        """Verifica estrazione riuscita con mock."""
        # Crea un file immagine fittizio
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image content")

        mock_response = json.dumps({
            "squadra_casa": "Roma",
            "squadra_trasf": "Lazio",
            "quota_1": 1.85,
            "quota_x": 3.60,
            "quota_2": 4.20,
            "confidence": "high",
        })

        with patch("src.ocr._run_vision_command") as mock_run:
            mock_run.return_value = (True, mock_response, "")
            result = extract_from_image_file(img_path)
            assert result.extraction_success is True
            assert result.squadra_casa == "Roma"


class TestExtractFromBytes:
    """Test per extract_from_bytes."""

    def test_creates_temp_file(self):
        """Verifica che venga creato un file temporaneo."""
        fake_bytes = b"fake image content"
        mock_response = json.dumps({
            "squadra_casa": "Inter",
            "squadra_trasf": "Napoli",
            "quota_1": 2.00,
            "quota_x": 3.30,
            "quota_2": 3.50,
            "confidence": "medium",
        })

        with patch("src.ocr._run_vision_command") as mock_run:
            mock_run.return_value = (True, mock_response, "")
            result = extract_from_bytes(fake_bytes, ".jpg")
            assert result.extraction_success is True


class TestExtractFromBase64:
    """Test per extract_from_base64."""

    def test_valid_base64(self):
        """Verifica decodifica base64 valida."""
        import base64
        fake_content = b"fake image"
        b64_data = base64.b64encode(fake_content).decode()

        mock_response = json.dumps({
            "squadra_casa": "Atalanta",
            "squadra_trasf": "Fiorentina",
            "quota_1": 1.90,
            "quota_x": 3.50,
            "quota_2": 3.80,
            "confidence": "high",
        })

        with patch("src.ocr._run_vision_command") as mock_run:
            mock_run.return_value = (True, mock_response, "")
            result = extract_from_base64(b64_data, "image/png")
            assert result.extraction_success is True

    def test_invalid_base64(self):
        """Verifica gestione base64 invalido."""
        result = extract_from_base64("not valid base64!!!")
        assert result.extraction_success is False


class TestRunVisionCommand:
    """Test per _run_vision_command."""

    def test_cli_not_found(self):
        """Verifica errore quando CLI non è trovato."""
        with patch("src.ocr._find_zai_cli") as mock_find:
            mock_find.return_value = None
            success, stdout, stderr = _run_vision_command("/fake/image.jpg")
            assert success is False
            assert "non trovato" in stderr.lower()

    def test_timeout(self):
        """Verifica gestione timeout."""
        with patch("src.ocr._find_zai_cli") as mock_find:
            with patch("src.ocr._find_bun") as mock_bun:
                with patch("src.ocr.subprocess.run") as mock_run:
                    import subprocess
                    mock_find.return_value = "/fake/cli.js"
                    mock_bun.return_value = "/fake/bun"
                    mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=1)

                    success, stdout, stderr = _run_vision_command("/fake/image.jpg")
                    assert success is False
                    assert "timeout" in stderr.lower()
