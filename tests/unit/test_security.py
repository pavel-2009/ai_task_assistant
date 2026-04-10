from app.core.security import hash_password, validate_password_strength, verify_password


def test_different_hashes_for_same_password():
    password = 'StrongPass1!'
    assert hash_password(password) != hash_password(password)


def test_verify_correct_password():
    password = 'StrongPass1!'
    assert verify_password(password, hash_password(password)) is True


def test_verify_wrong_password():
    assert verify_password('WrongPass1!', hash_password('StrongPass1!')) is False


def test_empty_password():
    hashed = hash_password('')
    assert verify_password('', hashed) is True


def test_password_complexity_all_criteria():
    assert validate_password_strength('Aa1!aaaa') is True


def test_password_no_uppercase():
    assert validate_password_strength('aa1!aaaa') is False


def test_password_no_lowercase():
    assert validate_password_strength('AA1!AAAA') is False


def test_password_no_digits():
    assert validate_password_strength('Aa!aaaaa') is False


def test_password_no_special_symbols():
    assert validate_password_strength('Aa1aaaaa') is False


def test_password_too_short():
    assert validate_password_strength('Aa1!a') is False


def test_password_minimum_length():
    assert validate_password_strength('Aa1!aaaa') is True
