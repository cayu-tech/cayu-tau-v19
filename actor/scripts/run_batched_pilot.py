"""Run the frozen pilot with its existing multi-tool-round option enabled.

This changes proposal batching only. Tool safety declarations, authorization,
execution ordering, task selection, scoring and deadlines remain unchanged.
"""

import sys


def enable_batching(arguments: list[str]) -> list[str]:
    return [*arguments, "--runtime-parallel-tools"]


def main():
    # Import from the frozen artifact containing this launcher.
    import run_pilot

    original = run_pilot.cli.main

    def configured_main():
        sys.argv = enable_batching(sys.argv)
        return original()

    run_pilot.cli.main = configured_main
    run_pilot.main()


if __name__ == "__main__":
    main()
