import unittest

from app.schemas import ChordSegment
from app.services.amt.basic_pitch import NoteEvent
from app.services.theory.musical_postprocessor import (
    apply_music_theory_rules,
    merge_temporal_clusters,
    remove_harmonic_duplicates,
)


class MusicalPostprocessorTests(unittest.TestCase):
    def test_remove_harmonic_duplicates_octave(self) -> None:
        events = [
            NoteEvent(start_time_s=0.0, end_time_s=0.5, pitch_midi=60, velocity=90, amplitude=1.0),
            NoteEvent(start_time_s=0.02, end_time_s=0.5, pitch_midi=72, velocity=80, amplitude=0.5),
        ]

        out = remove_harmonic_duplicates(events)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].pitch_midi, 60)

    def test_merge_temporal_clusters(self) -> None:
        events = [
            NoteEvent(start_time_s=0.0, end_time_s=0.05, pitch_midi=64, velocity=80, amplitude=0.5),
            NoteEvent(start_time_s=0.03, end_time_s=0.08, pitch_midi=64, velocity=90, amplitude=0.8),
            NoteEvent(start_time_s=0.06, end_time_s=0.12, pitch_midi=64, velocity=70, amplitude=0.6),
        ]

        out = merge_temporal_clusters(events, window_ms=80.0)

        self.assertEqual(len(out), 1)
        ev = out[0]
        self.assertAlmostEqual(ev.start_time_s, 0.0, places=3)
        self.assertAlmostEqual(ev.end_time_s, 0.12, places=3)
        self.assertEqual(ev.pitch_midi, 64)
        self.assertEqual(ev.velocity, 90)
        self.assertAlmostEqual(ev.amplitude, 0.8, places=3)

    def test_apply_music_theory_rules_dissonance(self) -> None:
        events = [
            NoteEvent(start_time_s=0.0, end_time_s=0.5, pitch_midi=60, velocity=90, amplitude=0.9),
            NoteEvent(start_time_s=0.01, end_time_s=0.5, pitch_midi=61, velocity=70, amplitude=0.3),
        ]
        chords = [ChordSegment(start=0.0, end=1.0, label="C", confidence=0.9)]

        out = apply_music_theory_rules(events, chords=chords, key_sig=None)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].pitch_midi, 60)


if __name__ == "__main__":
    unittest.main()
