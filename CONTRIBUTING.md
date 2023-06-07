## General guidelines

To contribute, fork the repository and send a pull request.

When submitting code, please make every effort to follow existing conventions and style in order to keep the code as readable as possible.

Where appropriate, please provide unit tests or integration tests. Unit tests should be pytest based tests and be added to `<project>/tests`.

Please make sure all tests pass before submitting a pull request. It is also good if you squash your commits and add the tags `#major` or `#minor` to the pull request title if need be, otherwise your pull request will be considered a patch bump.

## Contributing to MARLlib

We welcome contributions to MARLlib! If you're interested in contributing, here are some areas where we would appreciate your help:

### New Tasks

MARLlib aims to support a wide range of multi-agent reinforcement learning tasks. If you have a new task in mind that you think would be valuable to include, please consider contributing its implementation. You can add the new task code under the `marllib/envs` directory.

### New Algorithms

We are always looking to expand the collection of algorithms available in MARLlib. If you have developed a new multi-agent RL algorithm or would like to implement an existing one that is missing, we encourage you to contribute. You can add the new algorithm code under the `/marllib/marl/algos` directory.

### Bug fixes and enhancements

Contributions to fix bugs or enhance the existing functionality of MARLlib are highly appreciated. If you come across any issues or have ideas for improvements, please feel free to contribute your changes. You can submit bug fixes and enhancements as pull requests to the repository.

We appreciate all contributions and will review them promptly. Thank you for your interest in improving MARLlib!


## Testing the code locally

To test the code locally you need to install the dependencies for the library in the current environment. Additionally, you need to install the dependencies for testing. All of those dependencies can be installed with:

```
pip install -e .
```

To run the tests simply execute:

```
pytest -v --cov=marllib --cov-report html
```
