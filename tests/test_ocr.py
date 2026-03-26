"""Test per il modulo OCR di estrazione dati da screenshot."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

from src.ocr import (
    ExtractedData,
    _fallback_extraction,
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


class TestGetOpenaiApiKey:
    """Test per _get_openai_api_key."""

    def test_from_env_var(self):
        """Verifica recupero da variabile d'ambiente."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-123"}):
            assert _get_openai_api_key() == "sk-test-123"

    def test_missing_key(self):
        """Verifica None quando la chiave non è presente."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.ocr.os.environ.get", return_value=None),
        ):
            result = _get_openai_api_key()
            assert result is None


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
        assert "Inter" in result.squadra_casa
        assert "Milan" in result.squadra_trasf

    def test_extract_quotes(self):
        """Verifica estrazione quote."""
        response = "Quote: 1.85, 3.60, 4.20 per questa partita"
        result = _fallback_extraction(response, "test error")
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
        assert is_valid is True
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
            quota_1=0.50,
            quota_x=3.40,
            quota_2=100.0,
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

    def test_api_key_missing(self, tmp_path):
        """Verifica errore quando API key non è configurata."""
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"fake image content")

        with patch("src.ocr._get_openai_api_key", return_value=None):
            result = extract_from_image_file(img_path)
            assert result.extraction_success is False
            assert "OPENAI_API_KEY" in result.error_message

    def test_successful_extraction(self, tmp_path):
        """Verifica estrazione riuscita con mock."""
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

        with patch("src.ocr._call_openai_vision", return_value=mock_response):
            result = extract_from_image_file(img_path)
            assert result.extraction_success is True
            assert result.squadra_casa == "Roma"


class TestExtractFromBytes:
    """Test per extract_from_bytes."""

    def test_successful_extraction(self):
        """Verifica estrazione da bytes."""
        fake_bytes = b"fake image content"
        mock_response = json.dumps({
            "squadra_casa": "Inter",
            "squadra_trasf": "Napoli",
            "quota_1": 2.00,
            "quota_x": 3.30,
            "quota_2": 3.50,
            "confidence": "medium",
        })

        with patch("src.ocr._call_openai_vision", return_value=mock_response):
            result = extract_from_bytes(fake_bytes, ".jpg")
            assert result.extraction_success is True

    def test_api_key_missing(self):
        """Verifica errore quando API key non è configurata."""
        with patch("src.ocr._get_openai_api_key", return_value=None):
            result = extract_from_bytes(b"fake", ".jpg")
            assert result.extraction_success is False
            assert "OPENAI_API_KEY" in result.error_message


class TestExtractFromBase64:
    """Test per extract_from_base64."""

    def test_valid_base64(self):
        """Verifica decodifica base64 valida."""
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

        with patch("src.ocr._call_openai_vision", return_value=mock_response):
            result = extract_from_base64(b64_data, "image/png")
            assert result.extraction_success is True

    def test_invalid_base64(self):
        """Verifica gestione base64 invalido."""
        result = extract_from_base64("not valid base64!!!")
        assert result.extraction_success is False


class TestCallOpenaiVision:
    """Test per _call_openai_vision."""

    def test_raises_without_api_key(self):
        """Verifica che sollevi RuntimeError senza API key."""
        import pytest

        from src.ocr import _call_openai_vision

        with (
            patch("src.ocr._get_openai_api_key", return_value=None),
            pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
        ):
            _call_openai_vision("base64data", "image/png")

    def test_calls_openai_client(self):
        """Verifica che chiami correttamente il client OpenAI."""
        from src.ocr import _call_openai_vision

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"squadra_casa": "Roma"}'

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        with (
            patch("src.ocr._get_openai_api_key", return_value="sk-test"),
            patch("src.ocr.OpenAI", return_value=mock_client_instance) as mock_openai,
        ):
            result = _call_openai_vision("base64data", "image/png")
            assert result == '{"squadra_casa": "Roma"}'
            mock_openai.assert_called_once_with(api_key="sk-test")
            mock_client_instance.chat.completions.create.assert_called_once()
