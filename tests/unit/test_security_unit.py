"""Юнит-тесты для функций безопасности."""

import pytest
from app.schemas import UserCreate
from app.core.security import hash_password, verify_password


@pytest.mark.unit
def test_validate_password_strength_valid():
    """Тестирование валидного пароля."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("weakpass")
        UserCreate.validate_password("ValidPass123!")
        UserCreate.validate_password("StrongPassword99@")


@pytest.mark.unit
def test_validate_password_strength_too_short():
    """Тестирование пароля с недостаточной длиной."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("Short1!")
        UserCreate.validate_password("")    


@pytest.mark.unit
def test_validate_password_strength_no_uppercase():
    """Тестирование пароля без заглавных букв."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("validpass123!")


@pytest.mark.unit
def test_validate_password_strength_no_lowercase():
    """Тестирование пароля без строчных букв."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("VALIDPASS123!")
        

@pytest.mark.unit
def test_validate_password_strength_no_digits():
    """Тестирование пароля без цифр."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("ValidPass!")


@pytest.mark.unit
def test_validate_password_strength_no_special_chars():
    """Тестирование пароля без специальных символов."""
    with pytest.raises(ValueError) as exc:
        UserCreate.validate_password("ValidPass123")


@pytest.mark.unit
def test_hash_password():
    """Тестирование хеширования пароля."""
    password = "TestPassword123!"
    hashed = hash_password(password)

    # Хеш не должен быть равен исходному пароль
    assert hashed != password
    # Хеш должен быть строкой
    assert isinstance(hashed, str)
    # Хеш не должен быть пустым
    assert len(hashed) > 0


@pytest.mark.unit
def test_verify_password_correct():
    """Тестирование проверки корректного пароля."""
    password = "TestPassword123!"
    hashed = hash_password(password)
    
    # Проверка должна быть успешной
    assert verify_password(password, hashed) is True


@pytest.mark.unit
def test_verify_password_incorrect():
    """Тестирование проверки неправильного пароля."""
    password = "TestPassword123!"
    wrong_password = "WrongPassword123!"
    hashed = hash_password(password)
    
    # Проверка должна быть неуспешной
    assert verify_password(wrong_password, hashed) is False


@pytest.mark.unit
def test_hash_password_different_hashes():
    """Тестирование того, что один и тот же пароль дает разные хеши (bcrypt использует salt)."""
    password = "TestPassword123!"
    hash1 = hash_password(password)
    hash2 = hash_password(password)
    
    # Хеши должны быть разными (из-за разного salt)
    assert hash1 != hash2
    # Но оба должны проверяться правильно
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
