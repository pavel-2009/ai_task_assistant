import pytest
from pydantic import ValidationError

from app.schemas.task import TaskCreate, TaskUpdate
from app.schemas.user import UserCreate


def test_user_create_valid():
    model = UserCreate(username='valid_user', password='StrongPass1!')
    assert model.username == 'valid_user'


def test_username_too_long():
    with pytest.raises(ValidationError):
        UserCreate(username='x' * 21, password='StrongPass1!')


def test_username_empty():
    with pytest.raises(ValidationError):
        UserCreate(username='', password='StrongPass1!')


def test_password_too_short():
    with pytest.raises(ValidationError):
        UserCreate(username='user', password='Aa1!aaa')


def test_password_without_uppercase():
    with pytest.raises(ValidationError):
        UserCreate(username='user', password='weakpass1!')


def test_password_without_special_symbol():
    with pytest.raises(ValidationError):
        UserCreate(username='user', password='StrongPass1')


def test_task_create_valid():
    task = TaskCreate(title='Title', description='Description')
    assert task.title == 'Title'


def test_title_too_long():
    with pytest.raises(ValidationError):
        TaskCreate(title='x' * 21, description='Description')


def test_title_empty():
    with pytest.raises(ValidationError):
        TaskCreate(title='', description='Description')


def test_description_too_long():
    with pytest.raises(ValidationError):
        TaskCreate(title='Title', description='x' * 401)


def test_task_update_partial():
    task = TaskUpdate(title='New title')
    assert task.title == 'New title'


def test_task_update_empty():
    task = TaskUpdate()
    assert task.model_dump(exclude_none=True) == {}
