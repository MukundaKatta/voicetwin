"""Core data processor with validation and transformation."""
import time, logging, hashlib, json, re
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    success: bool
    data: Any
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ValidationRule:
    def __init__(self, name: str, check: Callable[[Any], bool], message: str):
        self.name = name
        self.check = check
        self.message = message

class Validator:
    """Configurable input validator."""
    def __init__(self):
        self.rules: List[ValidationRule] = []

    def add_rule(self, name: str, check: Callable[[Any], bool], message: str) -> "Validator":
        self.rules.append(ValidationRule(name, check, message))
        return self

    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        errors = []
        for rule in self.rules:
            try:
                if not rule.check(data):
                    errors.append(f"{rule.name}: {rule.message}")
            except Exception as e:
                errors.append(f"{rule.name}: validation error - {e}")
        return len(errors) == 0, errors

class DataProcessor:
    """Process data through validation, transformation, and output stages."""

    def __init__(self, name: str = "processor"):
        self.name = name
        self.validator = Validator()
        self._transforms: List[Callable] = []
        self._history: List[ProcessingResult] = []

    def add_transform(self, fn: Callable[[Any], Any]) -> "DataProcessor":
        self._transforms.append(fn)
        return self

    def process(self, data: Any) -> ProcessingResult:
        start = time.time()
        errors = []
        warnings = []

        # Validate
        is_valid, validation_errors = self.validator.validate(data)
        if not is_valid:
            errors.extend(validation_errors)
            return ProcessingResult(False, data, errors, [], (time.time()-start)*1000)

        # Transform
        current = data
        for i, transform in enumerate(self._transforms):
            try:
                current = transform(current)
            except Exception as e:
                errors.append(f"Transform {i} failed: {e}")
                return ProcessingResult(False, current, errors, warnings, (time.time()-start)*1000)

        elapsed = (time.time() - start) * 1000
        result = ProcessingResult(True, current, errors, warnings, elapsed,
                                 {"transforms_applied": len(self._transforms), "processor": self.name})
        self._history.append(result)
        return result

    def process_batch(self, items: List[Any]) -> List[ProcessingResult]:
        return [self.process(item) for item in items]

    @property
    def stats(self) -> Dict:
        total = len(self._history)
        success = sum(1 for r in self._history if r.success)
        return {"processor": self.name, "total_processed": total, "success_rate": round(success/max(1,total), 3),
                "avg_duration_ms": round(sum(r.duration_ms for r in self._history)/max(1,total), 2)}
