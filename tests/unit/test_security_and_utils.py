import numpy as np

from app.core.security import hash_password, validate_password_strength, verify_password
from app.utils.math_ops import cosine_similarity


def test_password_hash_and_verify():
    password = "StrongPass1!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("WrongPass1!", hashed)


def test_validate_password_strength():
    assert validate_password_strength("StrongPass1!")
    assert not validate_password_strength("weak")


def test_cosine_similarity_basic_case():
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([1.0, 0.0])

    assert cosine_similarity(vec1, vec2) == 1.0


def test_cosine_similarity_zero_vector():
    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 1.0])

    assert cosine_similarity(vec1, vec2) == 0.0


def test_cosine_similarity_shape_mismatch():
    vec1 = np.array([1.0])
    vec2 = np.array([1.0, 2.0])

    try:
        cosine_similarity(vec1, vec2)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
