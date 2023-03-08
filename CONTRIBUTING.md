## General guidelines

To contribute, fork the repository and send a pull request.

When submitting code, please make every effort to follow existing conventions and style in order to keep the code as readable as possible.

Where appropriate, please provide unit tests or integration tests. Unit tests should be pytest based tests and be added to <project>/tests.

Please make sure all tests pass before submitting a pull request. It is also good if you squash your commits and add the tags #major or #minor to the pull request title if need be, otherwise your pull request will be considered a patch bump.


## Testing the code locally

To test the code locally you need to install the dependencies for the library in the current environment. Additionally, you need to install the dependencies for testing. All of those dependencies can be installed with:

```
pip install -e .
```

To run the tests simply execute:

```
pytest -v --cov=marllib --cov-report html
```
