import lambdaprompt


def test_first_example():
    assert True


def test_call_lambdaprompt():
    assert lambdaprompt.prompt("test", lambda: 1) == 1
    assert True
