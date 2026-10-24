from pydantic import BaseModel, Field
from assistant.cv_generator import CV
import json


def test_schema():
    schema = CV.model_json_schema()
    # Check for patterns
    # In CV: name: str = Field(min_length=1, pattern=r"\S")
    # This should be in the generated schema.

    # Let's inspect the schema
    print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    test_schema()
