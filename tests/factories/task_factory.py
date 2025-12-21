"""Factory for creating test Task entities."""

from datetime import datetime, timezone
from typing import Any

import factory
from factory import Faker

from app.domain.entities.task import Task


class TaskFactory(factory.Factory):
    """
    Factory for creating test Task entities.

    This factory uses factory_boy to generate realistic test data
    for Task entities with sensible defaults and customizable attributes.

    Example:
        >>> task = TaskFactory()
        >>> task = TaskFactory(status="processing")
        >>> task = TaskFactory.build(uuid="custom-uuid")
    """

    class Meta:
        """Factory configuration."""

        model = Task

    uuid = factory.Sequence(lambda n: f"task-{n}")
    status = "pending"
    task_type = "transcription"
    ref_user_id = None
    ref_request_id = None
    result = None
    file_name = Faker("file_name", extension="mp3")
    temp_file_name = Faker("temp_file_name", extension="mp3")
    transcript_temp_file_name = Faker("transcript_temp_file_name", extension="json")
    diarization_temp_file_name = Faker("diarization_temp_file_name", extension="json")
    url = None
    audio_duration = Faker("pyfloat", min_value=1.0, max_value=600.0, right_digits=2)
    language = "en"
    task_params = None
    duration = None
    start_time = None
    end_time = None
    error = None
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))

    @classmethod
    def queued(cls, **kwargs: Any) -> Task:
        """Create a task in queued status with start time."""
        return cls(status="queued", start_time=datetime.now(timezone.utc), **kwargs)
    
    @classmethod
    def processing(cls, **kwargs: Any) -> Task:
        """Create a task in processing status with start time."""
        return cls(status="processing", start_time=datetime.now(timezone.utc), **kwargs)

    @classmethod
    def completed(cls, **kwargs: Any) -> Task:
        """Create a completed task with result and timing."""
        result = kwargs.pop("result", {"segments": [{"text": "Test transcription"}]})
        return cls(
            status="completed",
            result=result,
            duration=10.5,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            **kwargs,
        )

    @classmethod
    def failed(cls, **kwargs: Any) -> Task:
        """Create a failed task with error message."""
        error = kwargs.pop("error", "Test error message")
        return cls(
            status="failed",
            error=error,
            end_time=datetime.now(timezone.utc),
            **kwargs,
        )
