"""JSON logging primitives used by Django, Celery, and subprocess runners."""

from __future__ import annotations

import ast
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from api.observability.context import get_log_context

try:
    from pythonjsonlogger import jsonlogger

    _BaseJsonFormatter = jsonlogger.JsonFormatter
except ImportError:  # pragma: no cover - production images install python-json-logger.
    class _BaseJsonFormatter(logging.Formatter):
        def add_fields(
            self,
            log_record: dict[str, Any],
            record: logging.LogRecord,
            message_dict: dict[str, Any],
        ) -> None:
            log_record.update(message_dict)

        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {}
            self.add_fields(payload, record, {})
            return json.dumps(payload, default=str)

_DEFAULT_FIELDS: dict[str, Any] = {
    "request_id": None,
    "job_public_id": None,
    "celery_task_id": None,
    "method_key": None,
    "target": None,
}

_RESERVED_LOG_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__)


def _compact_values(values: list[str]) -> str | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return ",".join(cleaned)


def _literal_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _literal_args(raw: Any) -> tuple[Any, ...]:
    if isinstance(raw, tuple):
        return raw
    if isinstance(raw, list):
        return tuple(raw)
    if not isinstance(raw, str) or not raw:
        return ()
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _multi_prediction_method_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str | None, str | None]:
    targets_obj = kwargs.get("targets")
    methods_obj = kwargs.get("methods")

    if targets_obj is None and len(args) >= 2:
        targets_obj = args[1]
    if methods_obj is None and len(args) >= 3:
        methods_obj = args[2]

    if isinstance(targets_obj, str):
        targets = [targets_obj]
    elif isinstance(targets_obj, (list, tuple)):
        targets = [str(target) for target in targets_obj if target]
    else:
        targets = []

    methods = methods_obj if isinstance(methods_obj, dict) else {}
    method_keys = [str(methods.get(target, "")) for target in targets if methods.get(target)]
    return _compact_values(method_keys), _compact_values(targets)


def _context_from_celery_task_payload(record: logging.LogRecord) -> dict[str, Any]:
    data = getattr(record, "data", None)
    if not isinstance(data, dict):
        return {}

    task_name = str(data.get("name") or "")
    args = _literal_args(data.get("args"))
    kwargs = _literal_dict(data.get("kwargs"))

    context: dict[str, Any] = {
        "celery_task_id": data.get("id"),
        "job_public_id": kwargs.get("public_id") or (args[0] if args else None),
    }

    method_key = kwargs.get("method_key")
    target = kwargs.get("target")
    if task_name.endswith("run_prediction") and len(args) >= 3:
        method_key = method_key or args[1]
        target = target or args[2]
    elif task_name.endswith("run_multi_prediction"):
        inferred_method_key, inferred_target = _multi_prediction_method_context(args, kwargs)
        method_key = method_key or inferred_method_key
        target = target or inferred_target

    context["method_key"] = method_key
    context["target"] = target
    return {key: value for key, value in context.items() if value not in (None, "")}


class CorrelationFilter(logging.Filter):
    """Attach service metadata and contextvars correlation fields to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.service = os.environ.get("WEBKINPRED_SERVICE", "webkinpred")
        for key, default in _DEFAULT_FIELDS.items():
            if not hasattr(record, key):
                setattr(record, key, default)
        for key, value in _context_from_celery_task_payload(record).items():
            if getattr(record, key, None) in (None, ""):
                setattr(record, key, value)
        for key, value in get_log_context().items():
            setattr(record, key, value)
        return True


class JsonLogFormatter(_BaseJsonFormatter):
    """Stable JSON formatter with ISO timestamps and useful exception fields."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["message"] = record.getMessage()
        log_record["service"] = getattr(record, "service", os.environ.get("WEBKINPRED_SERVICE", "webkinpred"))

        for key, default in _DEFAULT_FIELDS.items():
            log_record.setdefault(key, getattr(record, key, default))

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
            log_record["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key in log_record:
                continue
            if key.startswith("_"):
                continue
            log_record[key] = value
