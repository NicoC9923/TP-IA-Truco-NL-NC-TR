"""Which round ends which generation, with at most one logged size change (ADR 0062 amendment).

Rounds are counted from the start of the run. Generations end every ``rounds_per_generation``
rounds. After ``with_change(round=C, ...)`` they end every ``new`` rounds counted from C, which must
itself be a generation end of the old size. With no change this is the arithmetic of a fixed size.
"""


class GenerationSchedule:
    def __init__(self, rounds_per_generation, change=None, info=None):
        if rounds_per_generation < 1:
            raise ValueError('rounds per generation must be positive')
        self.size = rounds_per_generation
        self.change = change  # (round, generation, new size) or None
        self.info = info  # what the log and checkpoint say about the change

    def with_change(self, *, round, rounds_per_generation, info=None):
        if self.change is not None:
            raise ValueError('The generation size can be changed only once')
        if rounds_per_generation < 1 or round < 1 or round % self.size:
            raise ValueError('The generation size can only change at a generation boundary')
        return GenerationSchedule(self.size, (round, round // self.size, rounds_per_generation), info)

    @property
    def current_size(self):
        return self.change[2] if self.change else self.size

    def _segment(self, rounds):
        if self.change is None or rounds <= self.change[0]:
            return 0, 0, self.size
        return self.change[0], self.change[1], self.change[2]

    def last_boundary(self, rounds):
        start, _, size = self._segment(rounds)
        return start + (rounds - start) // size * size

    def is_boundary(self, rounds):
        return self.last_boundary(rounds) == rounds

    def previous_boundary(self, rounds):
        """The generation end before ``rounds`` (itself a boundary or not)."""
        return self.last_boundary(rounds - 1) if rounds > 1 else 0

    def generation_at(self, rounds):
        """Generations completed after ``rounds`` rounds."""
        start, generation, size = self._segment(rounds)
        return generation + (rounds - start) // size

    def total_rounds(self, generations):
        if self.change is None or generations <= self.change[1]:
            return generations * self.size
        return self.change[0] + (generations - self.change[1]) * self.change[2]

    def record(self):
        change = self.change
        return {'rounds_per_generation': self.size,
                **({'change_round': change[0], 'change_generation': change[1], 'new_rounds_per_generation': change[2]}
                   if change else {}),
                **({'info': self.info} if self.info else {})}

    @classmethod
    def from_record(cls, record):
        schedule = cls(record['rounds_per_generation'])
        if 'change_round' in record:
            schedule = schedule.with_change(round=record['change_round'],
                                            rounds_per_generation=record['new_rounds_per_generation'],
                                            info=record.get('info'))
        return schedule
