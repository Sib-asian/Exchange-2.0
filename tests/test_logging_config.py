"""
Test per logging_config.py — Configurazione logging strutturato.
"""

import json
import logging

import pytest

from src.logging_config import (
    AnalysisLogger,
    JSONFormatter,
    engine_logger,
    get_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Test per il formattatore JSON."""

    def test_format_basic_record(self):
        """Il formattatore deve produrre JSON valido."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_format_with_extra_data(self):
        """Deve includere extra_data nel JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"key": "value", "number": 123}
        output = formatter.format(record)
        data = json.loads(output)

        assert "data" in data
        assert data["data"]["key"] == "value"
        assert data["data"]["number"] == 123

    def test_format_with_exception(self):
        """Deve includere exception info nel JSON."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]

    def test_serialize_float_precision(self):
        """Deve arrotondare i float a 6 decimali."""
        formatter = JSONFormatter()
        result = formatter._serialize({"pi": 3.141592653589793})
        assert result["pi"] == 3.141593

    def test_serialize_nested_dict(self):
        """Deve serializzare dict annidati."""
        formatter = JSONFormatter()
        result = formatter._serialize({"outer": {"inner": {"value": 1.23456789}}})
        assert result["outer"]["inner"]["value"] == 1.234568

    def test_serialize_list(self):
        """Deve serializzare liste."""
        formatter = JSONFormatter()
        result = formatter._serialize([1.1111111, 2.2222222, 3.3333333])
        assert result == [1.111111, 2.222222, 3.333333]


class TestSetupLogging:
    """Test per la configurazione del logging."""

    def test_setup_default_logging(self):
        """Deve configurare il logging con valori di default."""
        setup_logging()
        root_logger = logging.getLogger()

        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_debug_level(self):
        """Deve rispettare il livello specificato."""
        setup_logging(level=logging.DEBUG)
        root_logger = logging.getLogger()

        assert root_logger.level == logging.DEBUG

    def test_setup_json_output(self):
        """Deve usare JSONFormatter quando richiesto."""
        setup_logging(json_output=True)
        root_logger = logging.getLogger()

        # Trova l'handler e verifica il formatter
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)


class TestGetLogger:
    """Test per get_logger."""

    def test_returns_logger_with_correct_name(self):
        """Deve restituire un logger con il nome corretto."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"

    def test_returns_same_logger_for_same_name(self):
        """Deve restituire lo stesso logger per lo stesso nome."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")
        assert logger1 is logger2


class TestAnalysisLogger:
    """Test per il context manager AnalysisLogger."""

    def test_context_manager_logs_start_and_end(self, caplog):
        """Deve loggare inizio e fine dell'analisi."""
        with caplog.at_level(logging.INFO):
            with AnalysisLogger(match_id="12345", minute=45):
                pass

        # Deve avere almeno un log di inizio e uno di fine
        messages = [r.message for r in caplog.records]
        assert any("Analysis started" in m for m in messages)
        assert any("Analysis completed" in m for m in messages)

    def test_data_logging(self, caplog):
        """Deve loggare i dati durante l'analisi."""
        with caplog.at_level(logging.DEBUG):
            with AnalysisLogger(match_id="12345") as log:
                log.data("xG_calculated", {"home": 1.5, "away": 0.8})

        # Il dato deve essere registrato
        assert len(log.data_points) == 1
        assert log.data_points[0]["name"] == "xG_calculated"

    def test_warning_logging(self, caplog):
        """Deve loggare warning durante l'analisi."""
        with caplog.at_level(logging.WARNING):
            with AnalysisLogger(match_id="12345") as log:
                log.warning("Test warning", {"detail": "value"})

        # Deve avere almeno un warning
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_exception_handling(self, caplog):
        """Deve loggare errori se l'analisi fallisce."""
        with caplog.at_level(logging.ERROR):
            try:
                with AnalysisLogger(match_id="12345"):
                    raise ValueError("Test error")
            except ValueError:
                pass

        # Deve avere un log di errore
        assert any("Analysis failed" in r.message for r in caplog.records)


class TestPreconfiguredLoggers:
    """Test per i logger pre-configurati."""

    def test_engine_logger_exists(self):
        """engine_logger deve esistere."""
        assert engine_logger is not None
        assert engine_logger.name == "exchange.engine"

    def test_engine_logger_can_log(self, caplog):
        """engine_logger deve poter loggare messaggi."""
        with caplog.at_level(logging.INFO):
            engine_logger.info("Test message")

        assert len(caplog.records) >= 1
