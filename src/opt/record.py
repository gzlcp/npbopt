from dataclasses import dataclass


@dataclass
class Record:
    win: int
    lose: int
    draw: int

    @property
    def total(self) -> int:
        return self.win + self.lose + self.draw
