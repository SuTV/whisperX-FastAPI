"""This module contains the task management routes for the FastAPI application."""

import json
from typing import Union
from fastapi import APIRouter, Depends, BackgroundTasks, File, UploadFile
from pydantic import ValidationError as PydanticValidationError
import pandas as pd

from app.api.dependencies import (
    get_alignment_service,
    get_diarization_service,
    get_task_management_service,
    get_transcription_service,
    get_speaker_assignment_service,
    get_file_service
)
from app.api.mappers.task_mapper import TaskMapper
from app.api.schemas.task_schemas import TaskListResponse
from app.core.exceptions import TaskNotFoundError, FileValidationError, ValidationError
from app.core.logging import logger
from app.schemas import (
    Metadata, Response, Result, TaskParams, TaskStatus, TaskType, SpeechToTextProcessingParams, VADOptions, ASROptions, WhisperModelParams, AlignmentParams, DiarizationParams, Transcript,
    AlignedTranscription, DiarizationSegment
)
from app.services.task_management_service import TaskManagementService
from app.domain.services.transcription_service import ITranscriptionService
from app.domain.services.alignment_service import IAlignmentService
from app.domain.services.diarization_service import IDiarizationService
from app.domain.services.speaker_assignment_service import ISpeakerAssignmentService
from app.services import (
    process_audio_common,
    process_transcribe,
    process_alignment,
    process_diarize,
    process_speaker_assignment
)
from app.services.file_service import FileService
from app.transcript import filter_aligned_transcription
from app.audio import process_audio_file
from app.api.constants import (
    JSON_EXTENSION,
    TASK_SCHEDULED_LOG_FORMAT,
)

task_router = APIRouter()


@task_router.get("/task/all", tags=["Tasks Management"])
async def get_all_tasks_status(
    task_params: TaskParams = Depends(),
    service: TaskManagementService = Depends(get_task_management_service),
) -> TaskListResponse:
    """
    Retrieve the status of all tasks.

    Args:
        service: Task management service dependency.

    Returns:
        TaskListResponse: The status of all tasks.
    """
    logger.info("Retrieving status of all tasks")
    tasks = service.get_all_tasks(status=task_params.status, ref_user_id=task_params.ref_user_id, ref_request_id=task_params.ref_request_id)

    # Convert domain tasks to API DTOs using mapper
    task_summaries = [TaskMapper.to_summary(task) for task in tasks]

    return TaskListResponse(tasks=task_summaries)


@task_router.get("/task/{identifier}", tags=["Tasks Management"])
async def get_transcription_status(
    identifier: str,
    service: TaskManagementService = Depends(get_task_management_service),
) -> Result:
    """
    Retrieve the status of a specific task by its identifier.

    Args:
        identifier (str): The identifier of the task.
        service: Task management service dependency.

    Returns:
        Result: The status of the task.

    Raises:
        TaskNotFoundError: If the identifier is not found.
    """
    logger.info("Retrieving status for task ID: %s", identifier)
    task = service.get_task(identifier)

    if task is None:
        logger.error("Task ID not found: %s", identifier)
        raise TaskNotFoundError(identifier)

    logger.info("Status retrieved for task ID: %s", identifier)
    return Result(
        status=task.status,
        result=task.result,
        metadata=Metadata(
            task_type=task.task_type,
            task_params=task.task_params,
            language=task.language,
            file_name=task.file_name,
            url=task.url,
            duration=task.duration,
            audio_duration=task.audio_duration,
            start_time=task.start_time,
            end_time=task.end_time,
        ),
        error=task.error,
    )


@task_router.delete("/task/{identifier}/delete", tags=["Tasks Management"])
async def delete_task(
    identifier: str,
    service: TaskManagementService = Depends(get_task_management_service),
) -> Response:
    """
    Delete a specific task by its identifier.

    Args:
        identifier (str): The identifier of the task.
        service: Task management service dependency.

    Returns:
        Response: Confirmation message of task deletion.

    Raises:
        TaskNotFoundError: If the task is not found.
    """
    logger.info("Deleting task ID: %s", identifier)
    if service.delete_task(identifier):
        logger.info("Task deleted: ID %s", identifier)
        return Response(identifier=identifier, message="Task deleted")
    else:
        logger.error("Task not found: ID %s", identifier)
        raise TaskNotFoundError(identifier)


@task_router.post("/task/{identifier}/retry", tags=["Tasks Management"])
async def retry_task(
    background_tasks: BackgroundTasks,
    identifier: str,
    transcript: Union[UploadFile, None] = File(default=None),
    diarization: Union[UploadFile, None] = File(default=None),
    service: TaskManagementService = Depends(get_task_management_service),
    transcription_service: ITranscriptionService = Depends(get_transcription_service),
    alignment_service: IAlignmentService = Depends(get_alignment_service),
    diarization_service: IDiarizationService = Depends(get_diarization_service),
    speaker_service: ISpeakerAssignmentService = Depends(get_speaker_assignment_service),
    file_service: FileService = Depends(get_file_service),
) -> Response:
    """
    Retry a specific task by its identifier.

    Args:
        identifier (str): The identifier of the task.
        transcript (UploadFile): Uploaded aligned transcript file.
        diarization (UploadFile): Uploaded diarization result file.
        service: Task management service dependency.

    Returns:
        Response: Confirmation message of task retrying.

    Raises:
        TaskNotFoundError: If the task is not found.
    """
    logger.info("Retrying task ID: %s", identifier)
    task = service.update_task_status(identifier, update_data={"status": TaskStatus.queued})

    if task is not None and task.temp_file_name is not None:
        if task.task_params and "vad_options" in task.task_params:
            vad_options_params = VADOptions(**task.task_params["vad_options"])
        if task.task_params and "asr_options" in task.task_params:
            asr_options_params = ASROptions(**task.task_params["asr_options"])
        if task.task_params:
            model_params = WhisperModelParams(**{k: v for k, v in task.task_params.items() if k in WhisperModelParams.__fields__})
        if task.task_params:
            align_params = AlignmentParams(**{k: v for k, v in task.task_params.items() if k in AlignmentParams.__fields__})
        if task.task_params:
            diarize_params = DiarizationParams(**{k: v for k, v in task.task_params.items() if k in DiarizationParams.__fields__})
        
        if task.task_type == TaskType.full_process:
            audio_params = SpeechToTextProcessingParams(
                audio=process_audio_file(task.temp_file_name),
                identifier=identifier,
                vad_options=vad_options_params,
                asr_options=asr_options_params,
                whisper_model_params=model_params,
                alignment_params=align_params,
                diarization_params=diarize_params,
            )
            background_tasks.add_task(process_audio_common, audio_params)
            logger.info(TASK_SCHEDULED_LOG_FORMAT, identifier)
        elif task.task_type == TaskType.transcription:
            print(task.task_type)
            background_tasks.add_task(
                process_transcribe,
                process_audio_file(task.temp_file_name),
                identifier,
                model_params,
                asr_options_params,
                vad_options_params,
                transcription_service,
            )
            logger.info(TASK_SCHEDULED_LOG_FORMAT, identifier)
        elif task.task_type == TaskType.transcription_alignment:
            if transcript is None:
                logger.error("Transcript file is required for alignment task retry: ID %s", identifier)
                raise TaskNotFoundError(f"Transcript file is required for alignment task retry: ID {identifier}")
            
            # Validate transcript file
            if transcript.filename is None:
                raise FileValidationError(
                    filename="unknown", reason="Transcript filename is missing"
                )
            
            file_service.validate_file_extension(transcript.filename, {JSON_EXTENSION})

            try:
                # Read the content of the transcript file
                transcript_data = Transcript(**json.loads(transcript.file.read()))
            except PydanticValidationError as e:
                logger.error("Invalid JSON content in transcript file: %s", str(e))
                raise ValidationError(
                    message=f"Invalid JSON content in transcript file: {str(e)}",
                    code="INVALID_TRANSCRIPT_JSON",
                    user_message="The transcript file contains invalid JSON.",
                )
            
            device = "cpu"
            if task.task_params and "device" in task.task_params:
                device = task.task_params["device"]

            background_tasks.add_task(
                process_alignment,
                process_audio_file(task.temp_file_name),
                transcript_data.model_dump(),
                identifier,
                device,
                align_params,
                alignment_service,
            )
            logger.info(TASK_SCHEDULED_LOG_FORMAT, identifier)
        elif task.task_type == TaskType.diarization:
            device = "cpu"
            if task.task_params and "device" in task.task_params:
                device = task.task_params["device"]
            
            background_tasks.add_task(
                process_diarize,
                process_audio_file(task.temp_file_name),
                identifier,
                device,
                diarize_params,
                diarization_service,
            )
            logger.info(TASK_SCHEDULED_LOG_FORMAT, identifier)
        elif task.task_type == TaskType.combine_transcript_diarization:
            if transcript is None:
                logger.error("Aligned transcript file is required for combine transcript - diarization task retry: ID %s", identifier)
                raise TaskNotFoundError(f"Aligned transcript file is required for combine transcript - diarization task retry: ID {identifier}")
            if diarization is None:
                logger.error("Diarization result file is required for combine transcript - diarization task retry: ID %s", identifier)
                raise TaskNotFoundError(f"Diarization result file is required for combine transcript - diarization task retry: ID {identifier}")
            
            # Validate files
            if transcript.filename is None:
                raise FileValidationError(
                    filename="unknown", reason="Aligned transcript filename is missing"
                )
            if diarization.filename is None:
                raise FileValidationError(
                    filename="unknown", reason="Diarization result filename is missing"
                )
            
            file_service.validate_file_extension(transcript.filename, {JSON_EXTENSION})
            file_service.validate_file_extension(diarization.filename, {JSON_EXTENSION})

            try:
                # Read the content of the transcript file
                transcript_data = AlignedTranscription(**json.loads(transcript.file.read()))
                # removing words within each segment that have missing start, end, or score values
                transcript_data = filter_aligned_transcription(transcript_data)
            except PydanticValidationError as e:
                logger.error("Invalid JSON content in aligned transcript file: %s", str(e))
                raise ValidationError(
                    message=f"Invalid JSON content in aligned transcript file: {str(e)}",
                    code="INVALID_TRANSCRIPT_JSON",
                    user_message="The aligned transcript file contains invalid JSON.",
                )
            try:
                # Map JSON to list of models
                diarization_segments = []
                for item in json.loads(diarization.file.read()):
                    diarization_segments.append(DiarizationSegment(**item))
            except PydanticValidationError as e:
                logger.error("Invalid JSON content in diarization result file: %s", str(e))
                raise ValidationError(
                    message=f"Invalid JSON content in diarization result file: {str(e)}",
                    code="INVALID_DIARIZATION_JSON",
                    user_message="The diarization result file contains invalid JSON.",
                )
            
            background_tasks.add_task(
                process_speaker_assignment,
                pd.json_normalize([segment.model_dump() for segment in diarization_segments]),
                transcript_data.model_dump(),
                identifier,
                speaker_service,
            )
            logger.info(TASK_SCHEDULED_LOG_FORMAT, identifier)
    
    logger.info("Task retried: ID %s", identifier)
    return Response(identifier=identifier, message="Task retried")