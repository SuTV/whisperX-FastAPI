import json
import pandas as pd
from app.core.logging import logger
from app.services.task_management_service import TaskManagementService
from app.api.dependencies import get_container
from app.schemas import (
    TaskStatus, TaskType, SpeechToTextProcessingParams, VADOptions, ASROptions, WhisperModelParams, AlignmentParams, DiarizationParams, Transcript,
    AlignedTranscription, DiarizationSegment
)
from app.services.task_management_service import TaskManagementService
from app.services import (
    process_audio_common,
    process_transcribe,
    process_alignment,
    process_diarize,
    process_speaker_assignment
)
from app.transcript import filter_aligned_transcription
from app.audio import process_audio_file


def execute_expired_tasks() -> None:
    """
    Execute tasks that have been in the 'queued' or 'processing' state for too long.
    """
    # get too long processing tasks and too long queued tasks
    task_management_service: TaskManagementService = get_container().task_management_service()
    tasks = task_management_service.get_expired_tasks()
    if len(tasks) == 0:
        logger.info("No expired queued tasks found.")
        return
    
    for task in tasks:
        identifier = task.uuid
        logger.info(f"Processing expired task with UUID: {identifier}")

        if task is not None:
            try:
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
                    if task.temp_file_name is not None:
                        audio_params = SpeechToTextProcessingParams(
                            audio=process_audio_file(task.temp_file_name),
                            identifier=identifier,
                            vad_options=vad_options_params,
                            asr_options=asr_options_params,
                            whisper_model_params=model_params,
                            alignment_params=align_params,
                            diarization_params=diarize_params,
                        )
                        process_audio_common(audio_params)
                    else:
                        task_management_service.update_task_status(
                            identifier=identifier,
                            update_data={"status": TaskStatus.failed, "error": "Temp file name is missing."},
                        )
                elif task.task_type == TaskType.transcription:
                    if task.temp_file_name is not None:
                        process_transcribe(process_audio_file(task.temp_file_name), identifier, model_params, asr_options_params, vad_options_params, get_container().transcription_service())
                    else:
                        task_management_service.update_task_status(
                            identifier=identifier,
                            update_data={"status": TaskStatus.failed, "error": "Temp file name is missing."},
                        )
                elif task.task_type == TaskType.transcription_alignment:
                    if task.temp_file_name is not None and task.transcript_temp_file_name is not None:
                        transcript_data = None
                        # Read the content of the transcript file
                        with open(task.transcript_temp_file_name, 'r') as transcript:
                            transcript_data = Transcript(**json.loads(transcript.file.read()))
                        if transcript_data is not None:
                            device = "cpu"
                            if task.task_params and "device" in task.task_params:
                                device = task.task_params["device"]
                            process_alignment(
                                process_audio_file(task.temp_file_name),
                                transcript_data.model_dump(),
                                identifier,
                                device,
                                align_params,
                                get_container().alignment_service(),
                            )
                        else:
                            task_management_service.update_task_status(
                                identifier=identifier,
                                update_data={"status": TaskStatus.failed, "error": "Transcript data is invalid."},
                            )
                    else:
                        task_management_service.update_task_status(
                            identifier=identifier,
                            update_data={"status": TaskStatus.failed, "error": "Temp file name or transcript temp file name is missing."},
                        )
                elif task.task_type == TaskType.diarization:
                    if task.temp_file_name is not None:
                        device = "cpu"
                        if task.task_params and "device" in task.task_params:
                            device = task.task_params["device"]
                        process_diarize(
                            process_audio_file(task.temp_file_name),
                            identifier,
                            device,
                            diarize_params,
                            get_container().diarization_service(),
                        )
                    else:
                        task_management_service.update_task_status(
                            identifier=identifier,
                            update_data={"status": TaskStatus.failed, "error": "Temp file name is missing."},
                        )
                elif task.task_type == TaskType.combine_transcript_diarization:
                    if task.transcript_temp_file_name is not None and task.diarization_temp_file_name is not None:            
                        transcript_data = None
                        # Read the content of the transcript file
                        with open(task.transcript_temp_file_name, 'r') as transcript:
                            transcript_data = AlignedTranscription(**json.loads(transcript.file.read()))
                            # removing words within each segment that have missing start, end, or score values
                            transcript_data = filter_aligned_transcription(transcript_data)
                        diarization_segments = None
                        with open(task.diarization_temp_file_name, 'r') as diarization:
                            # Map JSON to list of models
                            diarization_segments = []
                            for item in json.loads(diarization.file.read()):
                                diarization_segments.append(DiarizationSegment(**item))
                        if transcript_data is not None and diarization_segments is not None:
                            process_speaker_assignment(
                                pd.json_normalize([segment.model_dump() for segment in diarization_segments]),
                                transcript_data.model_dump(),
                                identifier,
                                get_container().speaker_assignment_service(),
                            )
                        else:
                            task_management_service.update_task_status(
                                identifier=identifier,
                                update_data={"status": TaskStatus.failed, "error": "Transcript data or diarization segments data is invalid."},
                            )
                    else:
                        task_management_service.update_task_status(
                            identifier=identifier,
                            update_data={"status": TaskStatus.failed, "error": "Transcript temp file name or diarization temp file name is missing."},
                        )
            except Exception as e:
                logger.error(f"Error processing task {identifier}: {str(e)}")
                task_management_service.update_task_status(
                    identifier=identifier,
                    update_data={"status": TaskStatus.failed, "error": str(e)},
                )
