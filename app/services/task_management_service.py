"""Service for task management operations."""

from typing import Any

from app.core.logging import logger
from app.domain.entities.task import Task
from app.domain.repositories.task_repository import ITaskRepository


class TaskManagementService:
    """Service for managing task operations.

    This service handles all business logic for task management including
    creating, retrieving, updating, and deleting tasks.
    """

    def __init__(self, repository: ITaskRepository) -> None:
        """
        Initialize the task management service.

        Args:
            repository: Task repository for data persistence
        """
        self.repository = repository

    def create_task(self, task: Task) -> str:
        """
        Create a new task in the repository.

        Args:
            task: The domain task entity to create

        Returns:
            The UUID of the created task
        """
        logger.debug("Creating new task: %s", task.uuid)
        identifier = self.repository.add(task)
        logger.info("Task created with UUID: %s", identifier)
        return identifier

    def get_task(self, identifier: str) -> Task | None:
        """
        Retrieve a task by its identifier.

        Args:
            identifier: The UUID of the task to retrieve

        Returns:
            The task domain entity if found, None otherwise
        """
        logger.debug("Retrieving task with identifier: %s", identifier)
        task = self.repository.get_by_id(identifier)

        if task:
            logger.debug("Task found: %s", identifier)
        else:
            logger.debug("Task not found: %s", identifier)

        return task

    def get_all_tasks(self, status: str | None = None, ref_user_id: str | None = None, ref_request_id: str | None = None) -> list[Task]:
        """
        Retrieve all tasks from the repository.

        Args:
            status (str | None): Filter by task status
            ref_user_id (str | None): Filter by reference user ID
            ref_request_id (str | None): Filter by reference request ID

        Returns:
            List of all task domain entities
        """
        logger.debug("Retrieving all tasks")
        tasks = self.repository.get_all(status=status, ref_user_id=ref_user_id, ref_request_id=ref_request_id)
        logger.info("Retrieved %d tasks", len(tasks))
        return tasks

    def get_expired_tasks(self) -> list[Task]:
        """
        Retrieve all expired tasks from the repository.

        Returns:
            List of expired task domain entities
        """
        logger.debug("Retrieving expired tasks")
        tasks = self.repository.get_expired_tasks()
        logger.info("Retrieved %d expired tasks", len(tasks))
        return tasks

    def delete_task(self, identifier: str) -> bool:
        """
        Delete a task by its identifier.

        Args:
            identifier: The UUID of the task to delete

        Returns:
            True if the task was deleted, False if not found
        """
        logger.debug("Deleting task with identifier: %s", identifier)
        result = self.repository.delete(identifier)

        if result:
            logger.info("Task deleted successfully: %s", identifier)
        else:
            logger.warning("Task not found for deletion: %s", identifier)

        return result

    def update_task_status(self, identifier: str, update_data: dict[str, Any]) -> Task | None:
        """
        Update task status and related information.

        Args:
            identifier: The UUID of the task to update
            update_data: Dictionary of fields to update

        Returns:
            The task domain entity if found, None otherwise
        """
        logger.debug("Updating task %s with data: %s", identifier, update_data.keys())
        task = self.repository.update(identifier, update_data)
        logger.info("Task updated successfully: %s", identifier)

        if task:
            logger.debug("Task updated: %s", identifier)
        else:
            logger.debug("Task not updated: %s", identifier)

        return task
