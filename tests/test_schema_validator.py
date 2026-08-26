import pytest
from fleet.schema_validator import SchemaValidator, ValidationError


class TestValidationError:
    def test_to_dict(self):
        e = ValidationError("field", "message", "warning")
        d = e.to_dict()
        assert d["field"] == "field"
        assert d["severity"] == "warning"


class TestSchemaValidator:
    def test_init(self):
        v = SchemaValidator()
        assert v.get_registered_schemas() == []

    def test_register(self):
        v = SchemaValidator()
        v.register("test", {"name": {"type": "string", "required": True}})
        assert "test" in v.get_registered_schemas()

    def test_validate_valid(self):
        v = SchemaValidator()
        v.register("test", {"name": {"type": "string", "required": True}})
        errors = v.validate({"name": "hello"}, "test")
        assert len(errors) == 0

    def test_validate_missing_required(self):
        v = SchemaValidator()
        v.register("test", {"name": {"type": "string", "required": True}})
        errors = v.validate({}, "test")
        assert len(errors) == 1
        assert errors[0].field == "name"

    def test_validate_wrong_type(self):
        v = SchemaValidator()
        v.register("test", {"age": {"type": "integer"}})
        errors = v.validate({"age": "twenty"}, "test")
        assert len(errors) == 1
        assert "Expected integer" in errors[0].message

    def test_validate_number_range(self):
        v = SchemaValidator()
        v.register("test", {"age": {"type": "number", "min": 0, "max": 120}})
        errors = v.validate({"age": -5}, "test")
        assert len(errors) == 1
        assert "minimum" in errors[0].message

    def test_validate_string_pattern(self):
        v = SchemaValidator()
        v.register("test", {"email": {"type": "string", "pattern": r"^.*@.*$"}})
        errors = v.validate({"email": "not-email"}, "test")
        assert len(errors) == 1
        assert "pattern" in errors[0].message

    def test_validate_nested_object(self):
        v = SchemaValidator()
        v.register(
            "test",
            {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "required": True}},
                }
            },
        )
        errors = v.validate({"user": {"name": "hello"}}, "test")
        assert len(errors) == 0

    def test_validate_array_items(self):
        v = SchemaValidator()
        v.register("test", {"scores": {"type": "array", "items": {"type": "number"}}})
        errors = v.validate({"scores": [1, 2, "three"]}, "test")
        assert len(errors) == 1
        assert "scores[2]" in errors[0].field

    def test_is_valid(self):
        v = SchemaValidator()
        v.register("test", {"name": {"type": "string", "required": True}})
        assert v.is_valid({"name": "hello"}, "test") is True
        assert v.is_valid({}, "test") is False

    def test_validate_missing_schema(self):
        v = SchemaValidator()
        errors = v.validate({}, "missing")
        assert len(errors) == 1
        assert "not found" in errors[0].message

    def test_to_dict(self):
        v = SchemaValidator()
        v.register("a", {})
        d = v.to_dict()
        assert d["schemas"] == 1
