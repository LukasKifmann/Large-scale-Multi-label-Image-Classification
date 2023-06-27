from typing import Iterable, Iterator, TypeVar, Optional

T = TypeVar("T")


def prog(
    text: str, iter: Iterable[T], steps: Optional[int] = None, verbose: bool = False
) -> Iterator[T]:
    if not steps:
        steps = len(iter)  # type: ignore
    if steps:
        stepsstr = str(steps)
        progress = -1
        for i, x in enumerate(iter):
            new_progress = i * 100 // steps
            if verbose:
                progress = new_progress
                istr = str(i + 1)
                spaces = " " * (len(stepsstr) - len(istr))
                print(f"\r{text} {spaces}{istr}/{stepsstr} [{progress:3d}%]", end="")
            elif new_progress > progress:
                progress = new_progress
                print(f"\r{text} [{progress:3d}%]", end="")
            yield x
        if verbose:
            print(f"\r{text} {steps}/{steps} [100%]")
        else:
            print(f"\r{text} [100%]")
