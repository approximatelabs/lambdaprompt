Welcome! We are excited for your interest

Some principles
Pythonic
Simple
Hooks and context managers for configurability

Setting up a dev environment

Adding a new prompt to the library

Do not add new dependency libraries (trying to keep this as lightweight as possible)
If you do use one, try and import inside of the prompt (raising error here is fine)
It’s fine to depend on other prompts, but include a “signature-check”

See (x)

Adding a new api backend
Defaults
Kwargs for overriding defaults
Context manager for config overrides
Furthest outside wins

See (x)
