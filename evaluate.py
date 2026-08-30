from __future__ import annotations

import sys


COMMANDS = {
    "fcn_ocr": "fcn_ocr.evaluation.fcn_ocr",
    "vertical_segmentation": "fcn_ocr.evaluation.vertical_segmentation",
    "baseline_detection": "fcn_ocr.evaluation.baseline_detection",
}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        commands = "\n".join(f"  {name}" for name in COMMANDS)
        print(
            "Usage: python evaluate.py TASK [TASK OPTIONS]\n\n"
            "Tasks:\n"
            f"{commands}\n\n"
            "Run 'python evaluate.py TASK --help' for task-specific options."
        )
        return

    task = arguments.pop(0)
    try:
        module_name = COMMANDS[task]
    except KeyError as error:
        raise SystemExit(
            f"Unknown evaluation task {task!r}; expected one of {tuple(COMMANDS)}"
        ) from error

    if module_name.endswith(".fcn_ocr"):
        from fcn_ocr.evaluation.fcn_ocr import main as task_main
    elif module_name.endswith(".vertical_segmentation"):
        from fcn_ocr.evaluation.vertical_segmentation import main as task_main
    else:
        from fcn_ocr.evaluation.baseline_detection import main as task_main
    task_main(arguments)


if __name__ == "__main__":
    main()
